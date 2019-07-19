/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

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

#define LENGTH 4

#define SIZE (LENGTH * LENGTH)

#define THREADS_PER_BLOCK 1
#define BLOCKS_PER_GRID LENGTH

// CPU function - basically scan each row and save the output in array
void matrixRowSum(int* input, int* output, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            output[i] += input[i * width + j];
        }
    }
}

// Device (kernel) function
__global__ void gpuMatrixRowSum(int* input, int* output, int width) {
    int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
#pragma unroll
    for (int i = 0; i < width; i++) {
        output[index] += input[index * width + i];
    }
}

int main() {
    int* Matrix;
    int* sumMatrix;
    int* cpuSumMatrix;

    int* gpuMatrix;
    int* gpuSumMatrix;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    Matrix = (int*)malloc(sizeof(int) * SIZE);
    sumMatrix = (int*)malloc(sizeof(int) * LENGTH);
    cpuSumMatrix = (int*)malloc(sizeof(int) * LENGTH);

    for (int i = 0; i < SIZE; i++) {
        Matrix[i] = i * 2;
    }

    for (int i = 0; i < LENGTH; i++) {
        cpuSumMatrix[i] = 0;
    }

    // Allocated Device Memory
    hipMalloc((void**)&gpuMatrix, SIZE * sizeof(int));
    hipMalloc((void**)&gpuSumMatrix, LENGTH * sizeof(int));

    // Memory Copy to Device
    hipMemcpy(gpuMatrix, Matrix, SIZE * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(gpuSumMatrix, cpuSumMatrix, LENGTH * sizeof(float), hipMemcpyHostToDevice);

    // Launch device kernels
    hipLaunchKernelGGL(gpuMatrixRowSum, dim3(BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK), 0, 0,
                       gpuMatrix, gpuSumMatrix, LENGTH);

    // Memory copy back to device
    hipMemcpy(sumMatrix, gpuSumMatrix, LENGTH * sizeof(int), hipMemcpyDeviceToHost);

    // Cpu implementation
    matrixRowSum(Matrix, cpuSumMatrix, LENGTH);


    // verify the results
    int errors = 0;
    for (int i = 0; i < LENGTH; i++) {
        if (sumMatrix[i] != cpuSumMatrix[i]) {
            printf("%d - cpu: %d gpu: %d\n", i, sumMatrix[i], cpuSumMatrix[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("PASSED\n");
    } else {
        printf("FAILED with %d errors\n", errors);
    }

    // GPU Free
    hipFree(gpuMatrix);
    hipFree(gpuSumMatrix);

    // CPU Free
    free(Matrix);
    free(sumMatrix);
    free(cpuSumMatrix);

    return errors;
}
