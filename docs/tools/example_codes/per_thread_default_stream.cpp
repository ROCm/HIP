// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

int main()
{
    // Initialize the HIP runtime
    if (auto err = hipInit(0); err != hipSuccess)
    {
        std::cerr << "Failed to initialize HIP runtime." << std::endl;
        return EXIT_FAILURE;
    }

    // Get the per-thread default stream
    hipStream_t stream = hipStreamPerThread;

    // Use the stream for some operation
    // For example, allocate memory on the device
    void* d_ptr;
    std::size_t size = 1024;
    if (auto err = hipMalloc(&d_ptr, size); err != hipSuccess)
    {
        std::cerr << "Failed to allocate memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Perform some operation using the stream
    // For example, set memory on the device
    if (auto err = hipMemsetAsync(d_ptr, 0, size, stream); err != hipSuccess)
    {
        std::cerr << "Failed to set memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Synchronize the stream
    if (auto err = hipStreamSynchronize(stream); err != hipSuccess)
    {
        std::cerr << "Failed to synchronize stream." << std::endl;
        return EXIT_FAILURE;
    }

    // Free the allocated memory
    if(auto err = hipFree(d_ptr); err != hipSuccess)
    {
        std::cerr << "Failed to free memory." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Operation completed successfully using per-thread default stream." << std::endl;

    return EXIT_SUCCESS;
}
// [sphinx-end]
