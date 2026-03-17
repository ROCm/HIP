// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <iostream>

#define HIP_CHECK(expression)              \
{                                          \
    const hipError_t err = expression;     \
    if(err != hipSuccess)                  \
    {                                      \
        std::cerr << "HIP error: "         \
            << hipGetErrorString(err)      \
            << " at " << __LINE__ << "\n"; \
    }                                      \
}

// Addition of two values.
__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

// Declare a, b and c as static variables.
__managed__ int a, b, c;

int main()
{
    // Setup input values.
    a = 1;
    b = 2;

    // Launch add() kernel on GPU.
    add<<<1, 1>>>(&a, &b, &c);

    // Wait for GPU to finish before accessing on host.
    HIP_CHECK(hipDeviceSynchronize());

    // Print the result.
    std::cout << a << " + " << b << " = " << c << std::endl;

    return 0;
}
// [sphinx-end]
