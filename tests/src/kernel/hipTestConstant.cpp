/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include "test_common.h"

#define HIP_ASSERT(status) assert(status == hipSuccess)

#define LEN 512
#define SIZE 2048

__constant__ int Value[LEN];

__global__ void Get(int* Ad) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Ad[tid] = Value[tid];
}

int main() {
    int *A, *B, *Ad;
    A = new int[LEN];
    B = new int[LEN];
    for (unsigned i = 0; i < LEN; i++) {
        A[i] = -1 * i;
        B[i] = 0;
    }

    HIP_ASSERT(hipMalloc((void**)&Ad, SIZE));

    HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(Get, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);
    HIP_ASSERT(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

    for (unsigned i = 0; i < LEN; i++) {
        assert(A[i] == B[i]);
    }
    passed();
}
