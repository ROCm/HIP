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

int main()
{
    hipMemPool_t memPool;
    hipDevice_t device = 0; // Specify the device index.

    // Initialize the device.
    HIP_CHECK(hipSetDevice(device));

    // Get the default memory pool for the device.
    HIP_CHECK(hipDeviceGetDefaultMemPool(&memPool, device));

    // Allocate memory from the pool (e.g., 1 MB).
    std::size_t allocSize = 1 * 1024 * 1024;
    void* ptr;
    HIP_CHECK(hipMalloc(&ptr, allocSize));

    // Free the allocated memory.
    HIP_CHECK(hipFree(ptr));

    // Trim the memory pool to a specific size (e.g., 512 KB).
    std::size_t newSize = 512 * 1024;
    HIP_CHECK(hipMemPoolTrimTo(memPool, newSize));

    std::cout << "Memory pool trimmed to " << newSize << " bytes." << std::endl;
    return EXIT_SUCCESS;
}
// [sphinx-end]
