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

int main()
{
    int a, b, c;
    int *d_a, *d_b, *d_c;

    // Setup input values.
    a = 1;
    b = 2;

    // Allocate device copies of a, b and c.
    HIP_CHECK(hipMalloc(&d_a, sizeof(*d_a)));
    HIP_CHECK(hipMalloc(&d_b, sizeof(*d_b)));
    HIP_CHECK(hipMalloc(&d_c, sizeof(*d_c)));

    // Copy input values to device.
    HIP_CHECK(hipMemcpy(d_a, &a, sizeof(*d_a), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, &b, sizeof(*d_b), hipMemcpyHostToDevice));

    // Launch add() kernel on GPU.
    add<<<1, 1>>>(d_a, d_b, d_c);

    // Copy the result back to the host.
    HIP_CHECK(hipMemcpy(&c, d_c, sizeof(*d_c), hipMemcpyDeviceToHost));

    // Cleanup allocated memory.
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    // Prints the result.
    std::cout << a << " + " << b << " = " << c << std::endl;

    return 0;
}
// [sphinx-end]
