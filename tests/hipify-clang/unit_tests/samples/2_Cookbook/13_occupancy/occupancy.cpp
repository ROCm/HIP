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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// CHECK: #include "hip/hip_runtime.h"
#include "cuda_runtime.h"
#include <iostream>
#define NUM 1000000

// CHECK: if (status != hipSuccess) {
#define CUDA_CHECK(status)                                                                         \
    if (status != cudaSuccess) {                                                                   \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
        exit(0);                                                                                   \
    }

// Device (Kernel) function
__global__ void multiply(float* C, float* A, float* B, int N) {
    int tx = blockDim.x*blockIdx.x+threadIdx.x;
    if (tx < N) {
        C[tx] = A[tx] * B[tx];
    }
}

// CPU implementation
void multiplyCPU(float* C, float* A, float* B, int N) {
    for(unsigned int i=0; i<N; i++) {
        C[i] = A[i] * B[i];
    }
}

void launchKernel(float* C, float* A, float* B, bool manual) {
    // CHECK: hipDeviceProp_t devProp;
    cudaDeviceProp devProp;
    // CHECK: CUDA_CHECK(hipGetDeviceProperties(&devProp, 0));
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));

    // CHECK: hipEvent_t start, stop;
    cudaEvent_t start, stop;
    // CHECK: CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&start));
    // CHECK: CUDA_CHECK(hipEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&stop));
    float eventMs = 1.0f;
    const unsigned threadsperblock = 32;
    const unsigned blocks = (NUM/threadsperblock) + 1;

    int mingridSize = 0;
    int gridSize = 0;
    int blockSize = 0;

    if (manual) {
        blockSize = threadsperblock;
        gridSize  = blocks;
        std::cout << std::endl << "Manual Configuration with block size " << blockSize << std::endl;
    } else {
        // CHECK: CUDA_CHECK(hipOccupancyMaxPotentialBlockSize(&mingridSize, &blockSize, multiply, 0, 0));
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&mingridSize, &blockSize, multiply, 0, 0));
        std::cout << std::endl << "Automatic Configuation based on hipOccupancyMaxPotentialBlockSize " << std::endl;
        std::cout << "Suggested blocksize is " << blockSize << ", Minimum gridsize is " << mingridSize << std::endl;
        gridSize = (NUM/blockSize)+1;
    }

    // Record the start event
    // CHECK: CUDA_CHECK(hipEventRecord(start, NULL));
    CUDA_CHECK(cudaEventRecord(start, NULL));

    // Launching the Kernel from Host
    // CHECK: hipLaunchKernelGGL(multiply, dim3(gridSize), dim3(blockSize), 0, 0, C, A, B, NUM);
    multiply <<<gridSize , blockSize>>> (C, A, B, NUM);

    // Record the stop event
    // CHECK: CUDA_CHECK(hipEventRecord(stop, NULL));
    CUDA_CHECK(cudaEventRecord(stop, NULL));
    // CHECK: CUDA_CHECK(hipEventSynchronize(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // CHECK: CUDA_CHECK(hipEventElapsedTime(&eventMs, start, stop));
    CUDA_CHECK(cudaEventElapsedTime(&eventMs, start, stop));
    printf("kernel Execution time = %6.3fms\n", eventMs);

    // Calculate Occupancy
    int numBlock = 0;
    // CHECK: CUDA_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, multiply, blockSize, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, multiply, blockSize, 0));

    if(devProp.maxThreadsPerMultiProcessor) {
        std::cout << "Theoretical Occupancy is " << (double)numBlock* blockSize/devProp.maxThreadsPerMultiProcessor * 100 << "%" << std::endl;
    }
}

int main() {
    float *A, *B, *C0, *C1, *cpuC;
    float *Ad, *Bd, *C0d, *C1d;
    int errors=0;

    // Initialize the input data
    A  = (float*)malloc(NUM * sizeof(float));
    B  = (float*)malloc(NUM * sizeof(float));
    C0 = (float*)malloc(NUM * sizeof(float));
    C1 = (float*)malloc(NUM * sizeof(float));
    cpuC = (float*)malloc(NUM * sizeof(float));

    for(int i=0; i< NUM; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Allocate the memory on the device side
    // CHECK: CUDA_CHECK(hipMalloc((void**)&Ad, NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&Ad, NUM * sizeof(float)));
    // CHECK: CUDA_CHECK(hipMalloc((void**)&Bd, NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&Bd, NUM * sizeof(float)));
    // CHECK: CUDA_CHECK(hipMalloc((void**)&C0d, NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&C0d, NUM * sizeof(float)));
    // CHECK: CUDA_CHECK(hipMalloc((void**)&C1d, NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&C1d, NUM * sizeof(float)));

    // Memory transfer from host to device
    // CHECK: CUDA_CHECK(hipMemcpy(Ad,A,NUM * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ad,A,NUM * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK: CUDA_CHECK(hipMemcpy(Bd,B,NUM * sizeof(float), hipMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Bd,B,NUM * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch with manual/default block size
    launchKernel(C0d, Ad, Bd, 1);

    // Kernel launch with the block size suggested by cudaOccupancyMaxPotentialBlockSize
    launchKernel(C1d, Ad, Bd, 0);

    // Memory transfer from device to host
    // CHECK: CUDA_CHECK(hipMemcpy(C0,C0d, NUM * sizeof(float), hipMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C0,C0d, NUM * sizeof(float), cudaMemcpyDeviceToHost));
    // CHECK: CUDA_CHECK(hipMemcpy(C1,C1d, NUM * sizeof(float), hipMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C1,C1d, NUM * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU computation
    multiplyCPU(cpuC, A, B, NUM);

    // Verify the results
    double eps = 1.0E-6;

    for (int i = 0; i < NUM; i++) {
        if (std::abs(C0[i] - cpuC[i]) > eps) {
            errors++;
        }
    }

    if (errors != 0) {
        printf("\nManual Test FAILED: %d errors\n", errors);
        errors=0;
    } else {
        printf("\nManual Test PASSED!\n");
    }

    for (int i = 0; i < NUM; i++) {
        if (std::abs(C1[i] - cpuC[i]) > eps) {
            errors++;
        }
    }

    if (errors != 0) {
        printf("\n Automatic Test FAILED: %d errors\n", errors);
    } else {
        printf("\nAutomatic Test PASSED!\n");
    }

    // CHECK: CUDA_CHECK(hipFree(Ad));
    CUDA_CHECK(cudaFree(Ad));
    // CHECK: CUDA_CHECK(hipFree(Bd));
    CUDA_CHECK(cudaFree(Bd));
    // CHECK: CUDA_CHECK(hipFree(C0d));
    CUDA_CHECK(cudaFree(C0d));
    // CHECK: CUDA_CHECK(hipFree(C1d));
    CUDA_CHECK(cudaFree(C1d));

    free(A);
    free(B);
    free(C0);
    free(C1);
    free(cpuC);
}
