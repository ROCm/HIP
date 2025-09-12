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
// SOFTWARE.

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <iostream>

#define HIP_CHECK(expression)                        \
{                                                    \
    const hipError_t status = expression;            \
    if (status != hipSuccess)                        \
    {                                                \
        std::cerr << "HIP error " << status          \
                << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":"         \
                << __LINE__ << std::endl;            \
        std::exit(EXIT_FAILURE);                     \
    }                                                \
}

// Kernel to perform some computation on allocated memory.
__global__ void myKernel(int* data, std::size_t numElements)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numElements)
    {
        data[tid] = tid * 2;
    }
}

int main()
{
    // Allocate memory.
    constexpr std::size_t numElements = 1024;
    int* devData;
    HIP_CHECK(hipMalloc(&devData, numElements * sizeof(*devData)));

    // Launch the kernel to perform computation.
    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
    myKernel<<<gridSize, blockSize>>>(devData, numElements);

    // Copy data back to host.
    int* hostData = new int[numElements];
    HIP_CHECK(hipMemcpy(hostData, devData, numElements * sizeof(*devData), hipMemcpyDeviceToHost));

    // Print the array.
    for (std::size_t i = 0; i < numElements; ++i)
        std::cout << "Element " << i << ": " << hostData[i] << std::endl;

    // Free memory.
    HIP_CHECK(hipFree(devData));
    delete[] hostData;

    // Synchronize to ensure completion.
    HIP_CHECK(hipDeviceSynchronize());

    return EXIT_SUCCESS;
}
// [sphinx-end]
