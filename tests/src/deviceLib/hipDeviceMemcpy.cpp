#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "../test_common.h"


#define LEN 1024
#define SIZE LEN << 2

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */


__global__ void cpy(hipLaunchParm lp, uint32_t *Out, uint32_t *In)
{
    int tx = hipThreadIdx_x;
    memcpy(Out + tx, In + tx, sizeof(uint32_t));
}

__global__ void set(hipLaunchParm lp, uint32_t *ptr, uint8_t val, size_t size)
{
    int tx = hipThreadIdx_x;
    memset(ptr + tx, val, (sizeof(uint32_t)*(size/LEN)));
}

int main()
{
    uint32_t *A, *Ad, *B, *Bd;
    uint32_t *Val, *Vald;
    A = new uint32_t[LEN];
    B = new uint32_t[LEN];
    Val = new uint32_t;
    *Val = 0;
    for(int i=0;i<LEN;i++){
        A[i] = i;
        B[i] = 0;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);

    hipLaunchKernel(cpy, dim3(1), dim3(LEN), 0, 0, Bd, Ad);

    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    for(int i=LEN-16;i<LEN;i++){
        if(A[i]!=B[i]){
            return 0;
        }
    }
    hipLaunchKernel(set, dim3(1), dim3(LEN), 0, 0, Bd, 0x1, LEN);

    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    for(int i=LEN-16;i<LEN;i++){
        if(0x01010101!=B[i]){
            return 0;
        }
    }
 
    passed();
}
