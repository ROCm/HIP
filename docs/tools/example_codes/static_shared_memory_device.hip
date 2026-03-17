// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

// [sphinx-start]
__global__ void kernel()
{
  __shared__ int array[128];
  __shared__ double result;
}
// [sphinx-end]

int main()
{
    kernel<<<64, 512>>>();
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Success!" << std::endl;
    return EXIT_SUCCESS;
}
