/*
 * Copyright (c) 2015-Present Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include "test_common.h"
#define NUM 1000

#define HIP_ASSERT(status) assert(status == hipSuccess)

__global__ void Empty(int param1, int* param2) {}

__global__ void multiply(float* C, float* A, float* B, int N, float* D){

    int tx = blockDim.x*blockIdx.x+threadIdx.x;

    if (tx < N){
        C[tx] = A[tx] * B[tx];
    }
}

int main() {
    //test case for passing NULL argument to an empty kernel
    hipLaunchKernelGGL(Empty, dim3(1), dim3(1), 0, 0, NULL, NULL);
    hipDeviceSynchronize();

    //test case for passing NULL argument to a kernel
    float *A, *B, *C;
    float *Ad, *Bd, *Cd;
    int i, blockSize=64;

    A  = (float *)malloc(NUM * sizeof(float));
    B  = (float *)malloc(NUM * sizeof(float));
    C  = (float *)malloc(NUM * sizeof(float));

   for(i=0; i< NUM; i++){
       A[i] = i;
       B[i] = i;
   }

   HIP_ASSERT(hipMalloc((void**)&Ad, NUM * sizeof(float)));
   HIP_ASSERT(hipMalloc((void**)&Bd, NUM * sizeof(float)));
   HIP_ASSERT(hipMalloc((void**)&Cd, NUM * sizeof(float)));

   HIP_ASSERT(hipMemcpy(Ad,A,NUM * sizeof(float), hipMemcpyHostToDevice));
   HIP_ASSERT(hipMemcpy(Bd,B,NUM * sizeof(float), hipMemcpyHostToDevice));

   hipLaunchKernelGGL(multiply, dim3((NUM/blockSize)+1), dim3(blockSize), 0, 0, Cd, Ad, Bd, NUM, NULL);

   HIP_ASSERT(hipMemcpy(C,Cd,NUM * sizeof(float), hipMemcpyDeviceToHost));

   for (i=0; i < NUM; i++) {
   	assert(C[i]==A[i]*B[i]);
   }

   hipFree(Ad);
   hipFree(Bd);
   hipFree(Cd);

   free(A);
   free(B);
   free(C);

   passed();
}