// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess)                 \
    {                                        \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
    }                                        \
}

// GPU Kernels
__global__ void kernelA(double* arrayA, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size)
    {
        arrayA[x] += 1.0;
    }
}

__global__ void kernelB(double* arrayA, double* arrayB, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size)
    {
        arrayB[x] += arrayA[x] + 3.0;
    }
}

int main()
{
    constexpr int numOfBlocks = 1 << 20;
    constexpr int threadsPerBlock = 1024;
    constexpr int numberOfIterations = 50;
    // The array size smaller to avoid the relatively short kernel launch compared to memory copies
    constexpr std::size_t arraySize = 1U << 25;
    double *d_dataA;
    double *d_dataB;

    double initValueA = 0.0;
    double initValueB = 2.0;

    std::vector<double> vectorA(arraySize, initValueA);
    std::vector<double> vectorB(arraySize, initValueB);
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_dataA, arraySize * sizeof(*d_dataA)));
    HIP_CHECK(hipMalloc(&d_dataB, arraySize * sizeof(*d_dataB)));
    // Create streams
    hipStream_t streamA, streamB;
    HIP_CHECK(hipStreamCreate(&streamA));
    HIP_CHECK(hipStreamCreate(&streamB));
    for(unsigned int iteration = 0; iteration < numberOfIterations; iteration++)
    {
        // Stream 1: Host to Device 1
        HIP_CHECK(hipMemcpyAsync(d_dataA, vectorA.data(), arraySize * sizeof(*d_dataA), hipMemcpyHostToDevice, streamA));
        // Stream 2: Host to Device 2
        HIP_CHECK(hipMemcpyAsync(d_dataB, vectorB.data(), arraySize * sizeof(*d_dataB), hipMemcpyHostToDevice, streamB));
        // Stream 1: Kernel 1
        kernelA<<<numOfBlocks, threadsPerBlock, 0, streamA>>>(d_dataA, arraySize);
        // Wait for streamA finish
        HIP_CHECK(hipStreamSynchronize(streamA));
        // Stream 2: Kernel 2
        kernelB<<<numOfBlocks, threadsPerBlock, 0, streamB>>>(d_dataA, d_dataB, arraySize);
        // Stream 1: Device to Host 2 (after Kernel 1)
        HIP_CHECK(hipMemcpyAsync(vectorA.data(), d_dataA, arraySize * sizeof(*vectorA.data()), hipMemcpyDeviceToHost, streamA));
        // Stream 2: Device to Host 2 (after Kernel 2)
        HIP_CHECK(hipMemcpyAsync(vectorB.data(), d_dataB, arraySize * sizeof(*vectorB.data()), hipMemcpyDeviceToHost, streamB));
    }
    // Wait for all operations in both streams to complete
    HIP_CHECK(hipStreamSynchronize(streamA));
    HIP_CHECK(hipStreamSynchronize(streamB));
    // Verify results
    double expectedA = (double)numberOfIterations;
    double expectedB = initValueB + (3.0 * numberOfIterations) + (expectedA * (expectedA + 1.0)) / 2.0;
    bool passed = true;
    for(std::size_t i = 0; i < arraySize; ++i)
    {
        if(vectorA[i] != expectedA)
        {
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedA << " got " << vectorA[i] << " at index: " << i << std::endl;
            break;
        }
        if(vectorB[i] != expectedB)
        {
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedB << " got " <<  vectorB[i] << " at index: " << i << std::endl;
            break;
        }
    }

    if(passed)
    {
        std::cout << "Asynchronous execution completed successfully." << std::endl;
    }
    else
    {
        std::cerr << "Asynchronous execution failed." << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipStreamDestroy(streamA));
    HIP_CHECK(hipStreamDestroy(streamB));
    HIP_CHECK(hipFree(d_dataA));
    HIP_CHECK(hipFree(d_dataB));

    return EXIT_SUCCESS;
}
// [sphinx-end]
