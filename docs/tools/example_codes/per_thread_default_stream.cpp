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
