// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
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

int main()
{
    std::size_t stackSize;
    HIP_CHECK(hipDeviceGetLimit(&stackSize, hipLimitStackSize));
    std::cout << "Default stack size: " << stackSize << " bytes" << std::endl;

    // Set a new stack size
    std::size_t newStackSize = 1024 * 8; // 8 KiB
    HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, newStackSize));

    HIP_CHECK(hipDeviceGetLimit(&stackSize, hipLimitStackSize));
    std::cout << "Updated stack size: " << stackSize << " bytes" << std::endl;

    return EXIT_SUCCESS;
}
// [sphinx-end]
