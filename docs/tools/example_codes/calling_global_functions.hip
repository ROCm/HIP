// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
