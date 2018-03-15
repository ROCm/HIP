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

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"

#define NUM 1024

#define THREADS_PER_BLOCK_X 4

// Device (Kernel) function, it must be void
// hipLaunchParm provides the execution configuration
__global__ void vmac_asm(hipLaunchParm lp, float* out, float* in) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    asm volatile("v_mac_f32_e32 %0, %2, %3" : "=v"(out[i]) : "0"(out[i]), "v"(a), "v"(in[i]));
}

// CPU implementation of saxpy
void CPUReference(float* output, float* input) {
    for (unsigned int j = 0; j < NUM; j++) {
        output[j] = a * input[j] + output[j];
    }
}

int main() {
    float* VectorA;
    float* ResultVector;
    float* VectorB;

    float* gpuVector;
    float* gpuResultVector;

    const float a = 10.0f int i;
    int errors;

    VectorA = (float*)malloc(NUM * sizeof(float));
    ResultVector = (float*)malloc(NUM * sizeof(float));
    VectorB = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        VectorA[i] = (float)i * 10.0f;
        VectorB[i] = (float)i * 30.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuVector, NUM * sizeof(float));
    hipMalloc((void**)&gpuResultVector, NUM * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuVector, VectorA, NUM * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(gpuResultVector, VectorB, NUM * sizeof(float), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernel(vmac_asm, dim3(NUM / THREADS_PER_BLOCK_X), dim3(THREADS_PER_BLOCK_X), 0, 0,
                    gpuResultVector, gpuVector);

    // Memory transfer from device to host
    hipMemcpy(ResultVector, gpuResultVector, NUM * sizeof(float), hipMemcpyDeviceToHost);

    // CPU Result computation
    addCPUReference(VectorB, VectorA);

    // verify the results
    errors = 0;
    double eps = 1.0E-3;
    for (i = 0; i < NUM; i++) {
        if (std::abs(ResultVector[i] - VectorB[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    hipFree(gpuVector);
    hipFree(gpuResultVector);

    hipDeviceReset();

    // free the resources on host side
    free(VectorA);
    free(ResultVector);
    free(VectorB);

    return errors;
}
