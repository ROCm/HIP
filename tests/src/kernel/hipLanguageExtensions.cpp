/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// Collection of code to make sure that various features in the hip kernel language compile.

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"

#include <test_common.h>

#ifdef __HCC__
#include <hc.hpp>
#endif

// cudaA

// Simple tests for variable type qualifiers:
__device__ int deviceVar;

// TODO-HCC __constant__ not working yet.
__constant__ int constantVar1;

__constant__ __device__ int constantVar2;

// Test HOST space:
__host__ void foo() { printf("foo!\n"); }

__device__ __noinline__ int sum1_noinline(int a) { return a + 1; };
__device__ __forceinline__ int sum1_forceinline(int a) { return a + 1; };


__device__ __host__ float PlusOne(float x) { return x + 1.0; }

__global__ void MyKernel(const float* a, const float* b, float* c,
                         unsigned N) {
    // KERNELBEGIN;

    unsigned gid = threadIdx.x;
    if (gid < N) {
        c[gid] = a[gid] + PlusOne(b[gid]);
    }

    // KERNELEND;
}


void callMyKernel() {
    float *a, *b, *c;
    const unsigned blockSize = 256;
    unsigned N = blockSize;

    hipLaunchKernelGGL(MyKernel, dim3(N / blockSize), dim3(blockSize), 0, 0, a, b, c, N);
}


template <typename T>
__global__ void vectorADD(T __restrict__* A_d, T* B_d, T* C_d, size_t N) {
//    KERNELBEGIN;
#ifdef NOT_YET
    int a = __shfl_up(x, 1);
#endif

    float x = 1.0;
    float z = sin(x);
#ifdef NOT_YET
    float fastZ = __sin(x);
#endif

#ifdef __HCC__

    int b = threadIdx.x;
    int c;

    atomicAdd(&c, b);
#endif

    __syncthreads();


    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] + B_d[i];
    }

    //    KERNELEND;
}


int main() {
    printf("Hello world\n");
    passed();
}
