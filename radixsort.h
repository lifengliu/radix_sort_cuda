#ifndef __RADIXSORT_H__
#define __RADIXSORT_H__
#include"common.h"

#define R 16
#define LOGR 4
float radixsort_cpu(uint32_t *in,uint32_t *out,int size);
bool check_sorted_int(uint32_t *in,int size);
float radixsort_gpu(uint32_t *in,uint32_t *out,int size);

#endif
