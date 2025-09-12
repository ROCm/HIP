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

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

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

// [sphinx-kernel-start]
__global__ void kernel()
{
  long long int start = clock64();
  // kernel code
  long long int stop = clock64();
  long long int cycles = stop - start;
}
// [sphinx-kernel-end]

int main()
{
    int deviceId = 0;

    // [sphinx-query-start]
    int wallClkRate = 0; //in kilohertz
    HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, deviceId));
    // [sphinx-query-end]

    kernel<<<dim3{1, 1, 1}, dim3{32,1,1}>>>();
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Device's wall clock rate is " << wallClkRate << " kHz." << std::endl;

    return EXIT_SUCCESS;
}
