#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include "common.h"
#include"prefixscan.h"

void prefix_scan_cpu(uint32_t *in,uint32_t *out,int size)
{
    out[0]=0;
    for(int i=1;i<size;i++)
    {
        out[i]=out[i-1]+in[i-1];
    }
}
__global__ void prefix_post_scan(int size,uint32_t *in,uint32_t *carry)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int base=blockDim.x*2*bid;
    uint32_t c=carry[bid];
    in[base+tid]+=c;
    in[base+tid+blockDim.x]+=c;
}

__global__ void prefix_scan_gpu_block(int size,uint32_t *in,uint32_t *carry)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    __shared__ uint32_t  block[PREFIX_SCAN_BLOCK_SIZE*2];
    int base=blockDim.x*2*bid;
    block[tid]=in[base+tid];
    block[tid+blockDim.x]=in[base+tid+blockDim.x];
    __syncthreads();
    for(int i=1;i<=blockDim.x;i*=2)
    {
        int index=(tid+1)*i*2-1;
        if(index<2*blockDim.x)
            block[index]+=block[index-i];
        __syncthreads();
    }
    //post scan
    for(int i=blockDim.x/2;i>=1;i=i/2)
    {
        int index=(tid+1)*i*2-1;
        if(index+i<2*blockDim.x)
        {
            block[index+i]+=block[index];
        }
        __syncthreads();
    }

    if(tid==0)
    {
        carry[bid]=block[2*blockDim.x-1];
        in[base+tid]=0;
    }
    else
    {
        in[base+tid]=block[tid-1];
    }
    in[base+blockDim.x+tid]=block[tid+blockDim.x-1];
}

void prefix_scan_kernel_padded(int block_size,int num_blocks,int size,uint32_t * dev_in)
{
    uint32_t *dev_carry=NULL;
    uint32_t *dev_carry_carry=NULL;
    uint32_t *carry_carry=NULL;
    int paded_num_blocks=0;
    int num_num_blocks=0;
    paded_num_blocks=((num_blocks-1)/(2*block_size)+1)*2*block_size;
    cudaMalloc((void **)(&dev_carry),paded_num_blocks*sizeof(uint32_t));
    cudaMemset(dev_carry,0,paded_num_blocks*sizeof(uint32_t));
    if(num_blocks>1)
    {
        num_num_blocks=paded_num_blocks/(block_size*2);
        cudaMalloc((void **)(&dev_carry_carry),num_num_blocks*sizeof(uint32_t));
         carry_carry=(uint32_t *)calloc(num_num_blocks,sizeof(uint32_t));
    }
    //round 1
    prefix_scan_gpu_block<<<num_blocks,block_size>>>(size,dev_in,dev_carry);
    if(num_blocks>1)
    {
        //round 2
        prefix_scan_gpu_block<<<num_num_blocks,block_size>>>(num_blocks,dev_carry,dev_carry_carry);
        //round 3
        if(num_num_blocks>1)
        {
            cudaMemcpy(carry_carry,dev_carry_carry,
                num_num_blocks*sizeof(uint32_t),cudaMemcpyDeviceToHost);
            uint32_t tmp=0;
            for(int i=0;i<num_num_blocks;i++)
            {
                uint32_t val=tmp+carry_carry[i];
                carry_carry[i]=tmp;
                tmp=val;
            }
            cudaMemcpy(dev_carry_carry,carry_carry,
                num_num_blocks*sizeof(uint32_t),cudaMemcpyHostToDevice);
        }
        else
        {
            cudaMemset(dev_carry_carry,0,sizeof(uint32_t));
        }
        //post round 2
        prefix_post_scan<<<num_num_blocks,block_size>>>(num_blocks,dev_carry,dev_carry_carry);
    }
    else
    {
        cudaMemset(dev_carry,0,sizeof(uint32_t));
    }
    //post round 1
    prefix_post_scan<<<num_blocks,block_size>>>(size,dev_in,dev_carry);

    CUT_CHECK_ERROR("kernel error:");
    //free
    if(dev_carry!=NULL)
        cudaFree(dev_carry);
    if(dev_carry_carry!=NULL)
        cudaFree(dev_carry_carry);
    if(carry_carry!=NULL)
        free(carry_carry);
}

void prefix_scan_gpu(int size,uint32_t *in,uint32_t * out)
{
    uint32_t * dev_in=NULL;
    //padding
    int padded_size=((size-1)/(2*PREFIX_SCAN_BLOCK_SIZE)+1)
		*2*PREFIX_SCAN_BLOCK_SIZE;
    cudaMalloc((void **)(&dev_in),padded_size*sizeof(uint32_t));
    cudaMemset(dev_in,0,padded_size*sizeof(uint32_t));
    int block_size=PREFIX_SCAN_BLOCK_SIZE;
    int num_blocks=padded_size/(block_size*2);

    cudaMemcpy(dev_in,in,size*sizeof(uint32_t),cudaMemcpyHostToDevice);
    prefix_scan_kernel_padded(block_size,num_blocks,size,dev_in);
    cudaMemcpy(out,dev_in,size*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    //free
    if(dev_in!=NULL)
        cudaFree(dev_in);
}

