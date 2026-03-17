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

__device__ unsigned long long fibonacci(unsigned long long n)
{
    if (n == 0 || n == 1)
    {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

__global__ void kernel(unsigned long long n)
{
    unsigned long long result = fibonacci(n);
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x == 0)
        printf("%llu! = %llu \n", n, result);
}

int main()
{
    kernel<<<1, 1>>>(10);
    HIP_CHECK(hipDeviceSynchronize());

    // With -O0 optimization option hit the stack limit
    // kernel<<<1, 256>>>(2048);
    // HIP_CHECK(hipDeviceSynchronize());

    return EXIT_SUCCESS;
}
// [sphinx-end]
