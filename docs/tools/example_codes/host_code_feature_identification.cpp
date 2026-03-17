// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if (err != hipSuccess)                                   \
    {                                                        \
        std::cout << "HIP Error: " << hipGetErrorString(err) \
              << " at line " << __LINE__ << std::endl;       \
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
}

int main()
{
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    int device = 0; // Query first available GPU. Can be replaced with any
                    // integer up to, not including, deviceCount
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));

    std::cout << "The queried device ";
    if (deviceProp.arch.hasSharedInt32Atomics) // portable HIP feature query
        std::cout << "supports";
    else
        std::cout << "does not support";
    std::cout << " shared int32 atomic operations" << std::endl;

    return EXIT_SUCCESS;
}
// [sphinx-end]
