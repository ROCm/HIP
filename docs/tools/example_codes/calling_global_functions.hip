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

#include <iostream>

#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if(err != hipSuccess)                                    \
    {                                                        \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
            << " at " << __LINE__ << "\n";                   \
    }                                                        \
}

// Performs a simple initialization of an array with the thread's index variables.
// This function is only available in device code.
__device__ void init_array(float * const a, const unsigned int arraySize)
{
    // globalIdx uniquely identifies a thread in a 1D launch configuration.
    const int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread initializes a single element of the array.
    if(globalIdx < arraySize)
    {
        a[globalIdx] = globalIdx;
    }
}

// Rounds a value up to the next multiple.
// This function is available in host and device code.
__host__ __device__ constexpr int round_up_to_nearest_multiple(int number, int multiple)
{
    return (number + multiple - 1)/multiple;
}

__global__ void example_kernel(float * const a, const unsigned int N)
{
    // Initialize array.
    init_array(a, N);
    // Perform additional work:
    // - work with the array
    // - use the array in a different kernel
    // - ...
}

int main()
{
    constexpr int N = 100000000; // problem size
    constexpr int blockSize = 256; //configurable block size

    //needed number of blocks for the given problem size
    constexpr int gridSize = round_up_to_nearest_multiple(N, blockSize);

    float *a;
    // allocate memory on the GPU
    HIP_CHECK(hipMalloc(&a, sizeof(*a) * N));

    std::cout << "Launching kernel." << std::endl;
    example_kernel<<<dim3(gridSize), dim3(blockSize), 0/*example doesn't use shared memory*/, 0/*default stream*/>>>(a, N);
    // make sure kernel execution is finished by synchronizing. The CPU can also
    // execute other instructions during that time
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;

    HIP_CHECK(hipFree(a));
}
// [sphinx-end]
