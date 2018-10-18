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

__global__ void vadd_hip(const float* a, const float* b, float* c, int N) {
    int idx = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}


int main(int argc, char* argv[]) {
    int sizeElements = 1000000;
    size_t sizeBytes = sizeElements * sizeof(float);
    bool pass = true;

    // Allocate host memory
    float* A_h = (float*)malloc(sizeBytes);
    float* B_h = (float*)malloc(sizeBytes);
    float* C_h = (float*)malloc(sizeBytes);

    // Allocate device memory:
    float *A_d, *B_d, *C_d;
    hipMalloc(&A_d, sizeBytes);
    hipMalloc(&B_d, sizeBytes);
    hipMalloc(&C_d, sizeBytes);

    // Initialize host memory
    for (int i = 0; i < sizeElements; i++) {
        A_h[i] = 1.618f * i;
        B_h[i] = 3.142f * i;
    }

    // H2D Copy
    hipMemcpy(A_d, A_h, sizeBytes, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, sizeBytes, hipMemcpyHostToDevice);

    // Launch kernel onto default accelerator
    int blockSize = 256;                                      // pick arbitrary block size
    int blocks = (sizeElements + blockSize - 1) / blockSize;  // round up to launch enough blocks
    hipLaunchKernelGGL(vadd_hip, dim3(blocks), dim3(blockSize), 0, 0, A_d, B_d, C_d, sizeElements);

    // D2H Copy
    hipMemcpy(C_h, C_d, sizeBytes, hipMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < sizeElements; i++) {
        float ref = 1.618f * i + 3.142f * i;
        if (C_h[i] != ref) {
            printf("error:%d computed=%6.2f, reference=%6.2f\n", i, C_h[i], ref);
            pass = false;
        }
    };
    if (pass) printf("PASSED!\n");
}
