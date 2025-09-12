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

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

int main()
{
    // [sphinx-start]
    std::size_t elements = 1 << 20;
    std::size_t size_bytes = elements * sizeof(int);

    // allocate host and device memory
    int *host_pointer = new int[elements];
    int *device_input, *device_result;
    HIP_CHECK(hipMalloc(&device_input, size_bytes));
    HIP_CHECK(hipMalloc(&device_result, size_bytes));

    // copy from host to the device
    HIP_CHECK(hipMemcpy(device_input, host_pointer, size_bytes, hipMemcpyHostToDevice));

    // Use memory on the device, i.e. execute kernels

    // copy from device to host, to e.g. get results from the kernel
    HIP_CHECK(hipMemcpy(host_pointer, device_result, size_bytes, hipMemcpyDeviceToHost));

    // free memory when not needed any more
    HIP_CHECK(hipFree(device_result));
    HIP_CHECK(hipFree(device_input));
    delete[] host_pointer;
    // [sphinx-end]

    return EXIT_SUCCESS;
}
