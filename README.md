# cuda-radix-sort
> Radix sort implemented on CUDA for integers. 

## System requirements:
> NVCC 6.0+
> Nvidia GPU kapler or higher

## How to make the test bench:
	make

## APIs:
```C
float radixsort_gpu(uint32_t *in,uint32_t *out,int size):
```
	Sort the numbers in "in" array and put results in out array,
	"size" is the length of the input array. Return time consumed 
	in ms.
```C
bool check_sorted_int(uint32_t *in,int size): 
```
	Check wither or not an array is sorted.
```C
float radixsort_cpu(uint32_t *in,uint32_t *out,int size):
```
	Radix sort algorithm working on CPU.For performance comparison.	
	
## Example:
Edit example.cu:
```C
	#include<stdio.h>
	#include<stdlib.h>
	#include<assert.h>
	#include"radixsort.h"
	
	int main()
	{
	    int N=25165824;
	    uint32_t * in=(uint32_t *) calloc(N,sizeof(uint32_t));
	    uint32_t * out=(uint32_t *) calloc(N,sizeof(uint32_t));
	    //random gen data
	    for(int i=0;i<N;i++)
	        in[i]=(uint32_t)rand();
	    float gpu_time=radixsort_gpu(in,out,N);
	    assert(check_sorted_int(out,N)==true);
	    printf("%d numbers sorted in %f ms\n",N,gpu_time);
	    return 0;
	}
```
Compile it with:
	nvcc -O3 -arch=sm_35 -o example example.cu radixsort.cu prefixscan.cu


	

