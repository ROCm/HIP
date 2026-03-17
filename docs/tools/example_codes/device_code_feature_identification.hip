// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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

__global__ void test_kernel()
{
    // [sphinx-start]
//#if __CUDA_ARCH__ >= 130 // does not properly specify, what feature is required, not portable
#if __HIP_ARCH_HAS_DOUBLES__ == 1 // explicitly specifies, what feature is required, portable between AMD and NVIDIA GPUs
    // device code
#endif
    // [sphinx-end]

#if __HIP_ARCH_HAS_DOUBLES__ == 1
    printf("Device has double-precision support.\n");
#else
    printf("Device does not have double-precision support.\n");
#endif
}

int main()
{
    test_kernel<<<1, 1, 0, 0>>>();
    HIP_CHECK(hipDeviceSynchronize());

    return EXIT_SUCCESS;
}
