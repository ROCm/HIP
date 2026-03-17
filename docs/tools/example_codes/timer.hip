// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
