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


#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "gxxApi1.h"

#define len 1024 * 1024
#define size len * sizeof(float)

__global__ void Kern(hipLaunchParm lp, float* A) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    A[tx] += 1.0f;
}

int main() {
    float A[len];
    float* Ad;

    for (int i = 0; i < len; i++) {
        A[i] = 1.0f;
    }

    Ad = (float*)mallocHip(size);
    memcpyHipH2D(Ad, A, size);
    hipLaunchKernel(HIP_KERNEL_NAME(Kern), dim3(len / 1024), dim3(1024), 0, 0, Ad);
    memcpyHipD2H(A, Ad, size);
    for (int i = 0; i < len; i++) {
        assert(A[i] == 2.0f);
    }

    hipFree(Ad);
}
