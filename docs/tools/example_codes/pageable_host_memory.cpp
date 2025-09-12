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

#include <cstring>
#include <iostream>

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess)                   \
    {                                          \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

int main()
{
    const int element_number = 100;

    int *host_input, *host_output;
    // Host allocation
    host_input  = new int[element_number];
    host_output = new int[element_number];

    // Host data preparation
    for (int i = 0; i < element_number; i++) {
        host_input[i] = i;
    }
    std::memset(host_output, 0, element_number * sizeof(int));

    int *device_input, *device_output;

    // Device allocation
    HIP_CHECK(hipMalloc((int **)&device_input,  element_number * sizeof(int)));
    HIP_CHECK(hipMalloc((int **)&device_output, element_number * sizeof(int)));

    // Device data preparation
    HIP_CHECK(hipMemcpy(device_input, host_input, element_number * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_output, 0, element_number * sizeof(int)));

    // Run the kernel
    // ...

    HIP_CHECK(hipMemcpy(device_input, host_input, element_number * sizeof(int), hipMemcpyHostToDevice));

    // Free host memory
    delete[] host_input;
    delete[] host_output;

    // Free device memory
    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}
// [sphinx-end]
