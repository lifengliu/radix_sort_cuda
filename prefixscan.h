#ifndef __PREFIXSCAN_H__
#define __PREFIXSCAN_H__
#include"common.h"

#define PREFIX_SCAN_BLOCK_SIZE 512

void prefix_scan_cpu(uint32_t *in,uint32_t *out,int size);
void prefix_scan_gpu(int size,uint32_t *in,uint32_t * out);
__global__ void prefix_scan_gpu_block(int size,uint32_t *in,uint32_t *carry);
void prefix_scan_kernel_padded(int block_size,int num_blocks,int size,uint32_t * dev_in);

#endif
