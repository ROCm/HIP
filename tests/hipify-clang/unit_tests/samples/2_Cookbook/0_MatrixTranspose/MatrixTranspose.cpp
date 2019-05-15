// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
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

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int main() {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    // CHECK: hipDeviceProp_t devProp;
    cudaDeviceProp devProp;
    // CHECK: hipGetDeviceProperties(&devProp, 0);
    cudaGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    // CHECK: hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    cudaMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    // CHECK: hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
    cudaMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Memory transfer from host to device
    // CHECK: hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);
    cudaMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice);

    // Lauching kernel from host

    dim3 dimGrid(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y);
    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    // CHECK: hipLaunchKernelGGL(matrixTranspose, dim3(dimGrid), dim3(dimBlock), 0, 0, gpuTransposeMatrix, gpuMatrix, WIDTH);
    matrixTranspose <<<dimGrid, dimBlock>>>(gpuTransposeMatrix, gpuMatrix, WIDTH);

    // Memory transfer from device to host
    // CHECK: hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
    cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }

    // free the resources on device side
    // CHECK: hipFree(gpuMatrix);
    cudaFree(gpuMatrix);
    // CHECK: hipFree(gpuTransposeMatrix);
    cudaFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    return errors;
}
