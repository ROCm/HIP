// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

__host__ __device__ void call_func()
{
    #ifdef __HIP_DEVICE_COMPILE__
        printf("device\n");
    #else
        std::cout << "host" << std::endl;
    #endif
}

__global__ void test_kernel()
{
  call_func();
}

int main()
{
    test_kernel<<<1, 1, 0, 0>>>();
    if(auto err = hipDeviceSynchronize(); err != hipSuccess)
        std::cerr << "HIP error " << err << ": " << hipGetErrorString(err) << std::endl;

    call_func();
    return EXIT_SUCCESS;
}
// [sphinx-end]
