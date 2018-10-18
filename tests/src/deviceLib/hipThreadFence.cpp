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

#include <iostream>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include "test_common.h"

#define NUM 1024
#define SIZE NUM * sizeof(float)

__global__ void vAdd(float* In1, float* In2, float* In3, float* In4, float* Out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    In4[tid] = In1[tid] + In2[tid];
    __threadfence();
    In3[tid] = In3[tid] + In4[tid];
    __threadfence_block();
    Out[tid] = In4[tid] + In3[tid];
}

int main() {
    float* In1 = new float[1024];
    float* In2 = new float[1024];
    float* In3 = new float[1024];
    float* In4 = new float[1024];
    float* Out = new float[1024];

    for (uint32_t i = 0; i < 1024; i++) {
        In1[i] = 1.0f;
        In2[i] = 1.0f;
        In3[i] = 1.0f;
        In4[i] = 1.0f;
    }

    float *In1d, *In2d, *In3d, *In4d, *Outd;
    hipMalloc((void**)&In1d, SIZE);
    hipMalloc((void**)&In2d, SIZE);
    hipMalloc((void**)&In3d, SIZE);
    hipMalloc((void**)&In4d, SIZE);
    hipMalloc((void**)&Outd, SIZE);

    hipMemcpy(In1d, In1, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(In2d, In2, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(In3d, In3, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(In4d, In4, SIZE, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(vAdd, dim3(32, 1, 1), dim3(32, 1, 1), 0, 0, In1d, In2d, In3d, In4d, Outd);
    hipMemcpy(Out, Outd, SIZE, hipMemcpyDeviceToHost);
    assert(Out[10] == 2 * In1[10] + 2 * In2[10] + In3[10]);
    passed();
}
