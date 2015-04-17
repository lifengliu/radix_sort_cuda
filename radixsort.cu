#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include"common.h"
#include"radixsort.h"
#include"prefixscan.h"

#define RADIX_BLOCK_SIZE 512
#define LOG_CHUNK 10
__device__ void prefix_scan_gpu_block_char(uint16_t *block)
{
	int tid=threadIdx.x;
	for(int i=1;i<=RADIX_BLOCK_SIZE;i*=2)
	{
		int index=(tid+1)*i*2-1;
		if(index<2*RADIX_BLOCK_SIZE)
			block[index]+=block[index-i];
		__syncthreads();	
	}
	//post scan
	for(int i=RADIX_BLOCK_SIZE/2;i>=1;i=i/2)
	{
		int index=(tid+1)*i*2-1;
		if(index+i<2*RADIX_BLOCK_SIZE)
		{
			block[index+i]+=block[index];
		}	
		__syncthreads();
	}
}

__device__ uint32_t reduceInWarp_uint(uint32_t f, int idInWarp){
    for(int i = warpSize/2; i > 0; i /= 2){
        f += __shfl_xor(f, i, 32);
    }
    return f;
}
__global__ void radix_sort_write_back(uint32_t *glb_in,uint32_t *glb_out,int d,
		uint32_t *glb_hist,uint32_t *local_hist)
{
	int tid=threadIdx.x;
	int bid=blockIdx.x;
	int base=RADIX_BLOCK_SIZE*2*bid;
	__shared__ uint32_t g_hist[R];
	__shared__ uint32_t l_hist[R+1];
	if(tid==0)
		l_hist[0]=0;
	if(tid<R)
	{
		g_hist[tid]=glb_hist[tid*gridDim.x+bid];
		l_hist[tid+1]=local_hist[tid*gridDim.x+bid];
		for(int i=1;i<=R/2;i*=2)
		{
			int index=(tid+1)*i*2-1;
			if(index<R)
				l_hist[index+1]+=l_hist[index-i+1];
		}
		//post scan
		for(int i=R/4;i>=1;i=i/2)
		{
			int index=(tid+1)*i*2-1;
			if(index+i<R)
			{
				l_hist[index+i+1]+=l_hist[index+1];
			}	
		}
	}
	__syncthreads();
	uint32_t num=glb_in[tid+base];
	int t=(num>>(d*LOGR))&(R-1);
	int dt=g_hist[t]+(tid-l_hist[t]);
	glb_out[dt]=num;
	
	num=glb_in[tid+base+RADIX_BLOCK_SIZE];
	t=(num>>(d*LOGR))&(R-1);	
	dt=g_hist[t]+(tid+RADIX_BLOCK_SIZE-l_hist[t]);
	glb_out[dt]=num;
}
__global__ void split_sort_block(uint32_t *glb_in,int d,uint32_t *hist)
{
	int tid=threadIdx.x;
	int bid=blockIdx.x;
	int base=RADIX_BLOCK_SIZE*2*bid;
	__shared__ uint32_t block[RADIX_BLOCK_SIZE*2*2];//ping-pong buffer
	block[tid]=glb_in[base+tid];
	block[tid+RADIX_BLOCK_SIZE]=glb_in[base+tid+RADIX_BLOCK_SIZE];
	__syncthreads();
	__shared__ uint16_t flag[RADIX_BLOCK_SIZE*2];
	uint32_t * in=block;
	uint32_t * out=block;
	for(int bit=d*LOGR;bit<(d+1)*LOGR;bit++)
	{
		in =(bit%2==0)?block:(block+RADIX_BLOCK_SIZE*2);		
		out =(bit%2==0)?(block+RADIX_BLOCK_SIZE*2):block;
		flag[tid]=((in[tid]>>bit)&1)^1;
		flag[tid+RADIX_BLOCK_SIZE]=((in[tid+RADIX_BLOCK_SIZE]>>bit)&1)^1;
		__syncthreads();
		prefix_scan_gpu_block_char(flag);
		int f=(tid==0)?0:(int)flag[tid-1];
		int totalFalse=(int)flag[RADIX_BLOCK_SIZE*2-1];
		int dt=((in[tid]>>bit)&1)?(tid-f+totalFalse)
				:f;
		out[dt]=in[tid];
		dt=(in[tid+RADIX_BLOCK_SIZE]>>bit)&1
			?(tid+RADIX_BLOCK_SIZE-
				(int)flag[tid+RADIX_BLOCK_SIZE-1]+totalFalse)
			:(int)flag[tid+RADIX_BLOCK_SIZE-1];
		out[dt]=in[tid+RADIX_BLOCK_SIZE];
		__syncthreads();
	}
	//writeback
	glb_in[base+tid]=out[tid];
	glb_in[base+tid+RADIX_BLOCK_SIZE]=out[tid+RADIX_BLOCK_SIZE];
	//get hist
	//reuse in
	if(tid<32*R)
		in[tid]=0;
	__syncthreads();
	if(tid<32)//single warp
	{
		for(int i=tid;i<RADIX_BLOCK_SIZE*2;i+=32)
		{
			int t=(out[i]>>(d*LOGR))&(R-1);
			in[t*32+tid]++;
		}
	}
	__syncthreads();
	if(tid<32*R)
	{
		uint32_t v=in[tid];
		int idInWarp=tid%32;
		int warpId=tid/32;
		v=reduceInWarp_uint(v,idInWarp);
		if(idInWarp==0)
			hist[bid+warpId*gridDim.x]=v;
	}
} 
float radixsort_gpu(uint32_t *in,uint32_t *out,int size)
{
	int padded_size=((size-1)/(2*RADIX_BLOCK_SIZE)+1)*(2*RADIX_BLOCK_SIZE);
	uint32_t * dev_in;
	uint32_t * dev_out;
	uint32_t * dev_hist;
	uint32_t * dev_local_hist;
	int num_blocks=padded_size/(2*RADIX_BLOCK_SIZE);
	int hist_size=num_blocks*R;
	int padded_hist_size=((hist_size-1)/(2*RADIX_BLOCK_SIZE)+1)*(2*RADIX_BLOCK_SIZE);
	cudaMalloc((void **)(&dev_in),padded_size*sizeof(uint32_t));
	cudaMalloc((void **)(&dev_out),padded_size*sizeof(uint32_t));
	cudaMalloc((void **)(&dev_hist),padded_hist_size*sizeof(uint32_t));
	cudaMalloc((void **)(&dev_local_hist),padded_hist_size*sizeof(uint32_t));
	//padding
	cudaMemset((dev_in+(padded_size-2*RADIX_BLOCK_SIZE)),0xffffffff,2*RADIX_BLOCK_SIZE*sizeof(uint32_t));
	cudaMemset((dev_out+(padded_size-2*RADIX_BLOCK_SIZE)),0xffffffff,2*RADIX_BLOCK_SIZE*sizeof(uint32_t));
	cudaMemset((dev_hist+(padded_hist_size-2*RADIX_BLOCK_SIZE)),0,2*RADIX_BLOCK_SIZE*sizeof(uint32_t));
	int num_hist_blocks=padded_hist_size/(2*RADIX_BLOCK_SIZE);
	uint32_t * input=dev_in;
	uint32_t * output=dev_out;
	cudaMemcpy(input,in,size*sizeof(uint32_t),cudaMemcpyHostToDevice);

	float elapsedTime=0;
	
#ifdef TIMING
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif

#ifdef TIMING
	cudaEventRecord(start,0);
#endif
	for(int round=0;round<8;round++)
	{
		input=(round%2==0)?dev_in:dev_out;
		output=(round%2==0)?dev_out:dev_in;
		split_sort_block<<<num_blocks,RADIX_BLOCK_SIZE>>>(input,round,dev_hist);
		cudaMemcpy(dev_local_hist,dev_hist,padded_hist_size*sizeof(uint32_t),cudaMemcpyDeviceToDevice);
		prefix_scan_kernel_padded(RADIX_BLOCK_SIZE,num_hist_blocks,hist_size,dev_hist);
		radix_sort_write_back<<<num_blocks,RADIX_BLOCK_SIZE>>>(input,output,round,
			dev_hist,dev_local_hist);
	}

#ifdef TIMING
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
#endif

	cudaMemcpy(out,output,size*sizeof(uint32_t),
		cudaMemcpyDeviceToHost);
    CUT_CHECK_ERROR("kernel error:");

#ifdef TIMING
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	//free
	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_hist);
	cudaFree(dev_local_hist);
	return elapsedTime;
}
float radixsort_cpu(uint32_t *in,uint32_t *out,int size)
{
	int bucket[R];
	int num_iter=(sizeof(uint32_t)*8)/LOGR;
	uint32_t * temp=(uint32_t *)calloc(size,sizeof(uint32_t));
	memcpy(out,in,size*sizeof(uint32_t));
	uint32_t *b1;
	uint32_t *b2;	
	float elapsedTime=0;
	//num_iter=1;
#ifdef TIMING
	clock_t start=clock();
#endif
	for(int iter=0;iter<num_iter;iter++)
	{
		if(iter%2==0)
		{
			b1=out;
			b2=temp;
		}
		else
		{
			b1=temp;
			b2=out;
		}
			
		for(int i=0;i<R;i++)
			bucket[i]=0;
		for(int i=0;i<size;i++)
		{
			int d=(b1[i]>>(LOGR*iter))&(R-1);
			bucket[d]++;
		}
		//scan bucket
		int sum=0;
		for(int i=0;i<R;i++)
		{
			int val=bucket[i];
			bucket[i]=sum;
			sum+=val;
		}
		//permute
		for(int i=0;i<size;i++)
		{
			int d=(b1[i]>>(LOGR*iter))&(R-1);
			int p=bucket[d];
			b2[p]=b1[i];
			bucket[d]=p+1;
		}
	}
#ifdef TIMING
	clock_t end=clock();
	elapsedTime=float(end-start)/float(CLOCKS_PER_SEC);
	elapsedTime*=1000.0;
#endif
	if(num_iter%2==1)
		memcpy(out,temp,size*sizeof(uint32_t));
	//free
	free(temp);
	return elapsedTime;
}

bool check_sorted_int(uint32_t *in,int size)
{
	if(size==0) return true;
	for(int i=0;i<size;i++)
	{
		if(in[i]<in[i-1])
		{
			return false;
		}	
	}
	return true;
}
