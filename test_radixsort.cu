#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include "common.h"
#include"radixsort.h"
#include"prefixscan.h"

float test_radix_cpu()
{
	int N=1024;
	uint32_t * in=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * out=(uint32_t *) calloc(N,sizeof(uint32_t));
	//gen data
	for(int i=0;i<N;i++)
		in[i]=(uint32_t)rand();
	float time=radixsort_cpu(in,out,N);
	if(check_sorted_int(out,N)==false)
		printf("FAILED\n");
	else
		printf("PASSED\n");	
	//free
	free(in);
	free(out);
	return time;
	
}
void test_radix_gpu(int N)
{
	//int N=1024*1024;
	uint32_t * in=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * out=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * cpu_out=(uint32_t *) calloc(N,sizeof(uint32_t));
	float cpu_time=0.0;
	float gpu_time=0.0;
	//gen data
	for(int i=0;i<N;i++)
		in[i]=(uint32_t)rand();
	cpu_time=radixsort_cpu(in,cpu_out,N);
	gpu_time=radixsort_gpu(in,out,N);
	for(int i=0;i<N;i++)
	{
		if(cpu_out[i]!=out[i])
		{
			printf("failed @%d expected=%u,got=%u\n",i,cpu_out[i],out[i]);
			return;
		}
	}
	if(check_sorted_int(out,N)==false)
		printf("FAILED\n");
	else
		printf("PASSED\n");	
#ifdef TIMING
	printf("gpu time= %f ms\n",gpu_time);
	printf("cpu time= %f ms\n",cpu_time);
	printf("speedup=%f ms\n",cpu_time/gpu_time);
#endif
	//free
	free(in);
	free(out);
	free(cpu_out);
}
	
void test_prefix_gpu(int N)
{
	//int N=1024*1024*24;
	//int N=735;
	//int N=797759;
	//int N=13468221;
	uint32_t * in=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * out=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * cpu_out=(uint32_t *) calloc(N,sizeof(uint32_t));
	
	//gen data
	for(int i=0;i<N;i++)
		in[i]=rand()%10;
	
	prefix_scan_gpu(N,in,out);
	prefix_scan_cpu(in,cpu_out,N);
	//check output
	for(int i=0;i<N;i++)
	{
		if(cpu_out[i]!=out[i])
		{
			printf("failed @%d, expected=%u, got=%u\n",i,cpu_out[i],out[i]);
			return;
		}
	}
	//for(int i=N-10;i<N;i++)
	//	printf("%u,%u\n",out[i],cpu_out[i]);
	//free

	free(in);
	free(out);
	free(cpu_out);
	printf("PASSED\n");
}
void test_split_sort_block()
{
	int N=1024;
	uint32_t * in=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * out=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * cpu_out=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t *dev_in;
	uint32_t *dev_hist;
	
	
	//gen data
	for(int i=0;i<N;i++)
		in[i]=rand()%16;
	//malloc
	cudaMalloc((void **)(&dev_in),N*sizeof(uint32_t));
	cudaMemcpy(dev_in,in,N*sizeof(uint32_t),cudaMemcpyHostToDevice);
	int block_size=512;
	int num_blocks=1;
	cudaMalloc((void **)(&dev_hist),num_blocks*R*sizeof(uint32_t));
	uint32_t * hist=(uint32_t *)calloc(num_blocks*R,
			sizeof(uint32_t));
	uint32_t * cpu_hist=(uint32_t *)calloc(num_blocks*R,
			sizeof(uint32_t));
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	split_sort_block<<<num_blocks,block_size>>>(dev_in,0,dev_hist);
	cudaMemcpy(out,dev_in,N*sizeof(uint32_t),
			cudaMemcpyDeviceToHost);	
	cudaMemcpy(hist,dev_hist,num_blocks*R*sizeof(uint32_t),
			cudaMemcpyDeviceToHost);
	radixsort_cpu(in,cpu_out,N);
	//check
	printf("testing out...\n");
	for(int i=0;i<N;i++)
	{
		if(cpu_out[i]!=out[i])
		{
			printf("failed @%d,expected=%u,got=%u\n",i,
				cpu_out[i],out[i]);
			return;
		}
	}
	printf("testing hist...\n");
	for(int i=0;i<N;i++)
	{
		cpu_hist[in[i]]++;
	}
	for(int i=0;i<R;i++)
	{
		if(cpu_hist[i]!=hist[i])
		{
			printf("failed @%d,expected=%u,got=%u\n",i,
				cpu_hist[i],hist[i]);
			return;
		}
	}

	CUT_CHECK_ERROR("kernel error:");
	//free
	cudaFree(dev_in);
	cudaFree(dev_hist);
	free(in);
	free(out);
	free(cpu_out);
	printf("PASSED\n");
}
void test_prefix_gpu_block()
{
	int N=1024;
	uint32_t * in=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * out=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * cpu_out=(uint32_t *) calloc(N,sizeof(uint32_t));
	uint32_t * dev_in;
	uint32_t * dev_carry;
	uint32_t * carry;
	//gen data
	for(int i=0;i<N;i++)
		in[i]=rand()%100;
	//malloc
	int block_size=512;
	int num_blocks=(N-1)/(block_size*2)+1;
	cudaMalloc((void **)(&dev_in),N*sizeof(uint32_t));
	cudaMalloc((void **)(&dev_carry),num_blocks*sizeof(uint32_t));
	carry=(uint32_t *)calloc(N,sizeof(uint32_t));
	//mem copy
	cudaMemcpy(dev_in,in,N*sizeof(uint32_t),cudaMemcpyHostToDevice);
	prefix_scan_gpu_block<<<num_blocks,block_size>>>(N,dev_in,dev_carry);
	cudaMemcpy(out,dev_in,N*sizeof(uint32_t),cudaMemcpyDeviceToHost);
	cudaMemcpy(carry,dev_carry,num_blocks*sizeof(uint32_t),cudaMemcpyDeviceToHost);

	prefix_scan_cpu(in,cpu_out,N);
	//check output
	for(int i=0;i<N;i++)
	{
		if(cpu_out[i]!=out[i])
		{
			printf("failed @%d, expected=%u, got=%u\n",i,cpu_out[i],out[i]);
			return;
		}
	}
	if(carry[0]!=(in[1023]+cpu_out[1023]))
	{
		printf("carry error");
		return;
	}
	
	CUT_CHECK_ERROR("kernel error:");
	//free
	cudaFree(dev_in);
	cudaFree(dev_carry);
	free(in);
	free(out);
	free(cpu_out);
	free(carry);
	printf("PASSED\n");
}
void do_test()
{
	printf("************test 1: test cpu radix sort\n");
	test_radix_cpu();		
	printf("************test 2: test gpu prefix scan block\n");
	test_prefix_gpu_block();		

	int data[]={735,1024,797759,1048576,13468221,25165824};
	for(int i=0;i<6;i++)
	{
		printf("************test 3: test gpu prefix scan N=%d\n",data[i]);
		test_prefix_gpu(data[i]);
	}

	printf("************test 4: test split sort block\n");
	test_split_sort_block();

	for(int i=0;i<6;i++)
	{
		printf("************test 5: test radix sort N=%d\n",data[i]);
		test_radix_gpu(data[i]);
	}
}

int main(int argc,char * argv[])
{
	do_test();	
	return 0;
}
