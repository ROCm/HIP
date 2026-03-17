// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
    // Create a stream.
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Create a memory pool with default properties.
    hipMemPoolProps poolProps = {};
    poolProps.allocType = hipMemAllocationTypePinned;
    poolProps.handleTypes = hipMemHandleTypePosixFileDescriptor;
    poolProps.location.type = hipMemLocationTypeDevice;
    poolProps.location.id = 0; // Assuming device 0.

    hipMemPool_t memPool;
    HIP_CHECK(hipMemPoolCreate(&memPool, &poolProps));

    // Allocate memory from the pool asynchronously.
    constexpr std::size_t numElements = 1024;
    int* devData = nullptr;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&devData),
                                     numElements * sizeof(*devData),
                                     memPool,
                                     stream));

    // Define grid and block sizes.
    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);

    // Launch the kernel to perform computation.
    myKernel<<<gridSize, blockSize, 0, stream>>>(devData, numElements);

    // Synchronize the stream.
    HIP_CHECK(hipStreamSynchronize(stream));

    // Copy data back to host.
    int* hostData = new int[numElements];
    HIP_CHECK(hipMemcpy(hostData, devData, numElements * sizeof(*devData), hipMemcpyDeviceToHost));

    // Print the array.
    for (std::size_t i = 0; i < numElements; ++i)
        std::cout << "Element " << i << ": " << hostData[i] << std::endl;

    // Free the allocated memory.
    HIP_CHECK(hipFreeAsync(devData, stream));

    // Synchronize the stream again to ensure all operations are complete.
    HIP_CHECK(hipStreamSynchronize(stream));

    // Destroy the memory pool and stream.
    HIP_CHECK(hipMemPoolDestroy(memPool));
    HIP_CHECK(hipStreamDestroy(stream));

    // Free host memory.
    delete[] hostData;

    return EXIT_SUCCESS;
}
// [sphinx-end]
