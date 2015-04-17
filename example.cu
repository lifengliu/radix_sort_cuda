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
