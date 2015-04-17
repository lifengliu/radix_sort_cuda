#ifndef __COMMON_H__
#define __COMMON_H__
#define TIMING 

#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

typedef unsigned long int uint64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;

#endif
