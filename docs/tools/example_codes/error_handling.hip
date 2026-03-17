// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess)                   \
    {                                          \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

// Addition of two values.
__global__ void add(int *a, int *b, int *c, std::size_t size)
{
    const std::size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < size)
    {
        c[index] += a[index] + b[index];
    }
}

int main()
{
    constexpr int numOfBlocks = 256;
    constexpr int threadsPerBlock = 256;
    constexpr std::size_t arraySize = 1U << 16;

    std::vector<int> a(arraySize), b(arraySize), c(arraySize);
    int *d_a, *d_b, *d_c;

    // Setup input values.
    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 2);

    // Allocate device copies of a, b and c.
    HIP_CHECK(hipMalloc(&d_a, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_c, arraySize * sizeof(int)));

    // Copy input values to device.
    HIP_CHECK(hipMemcpy(d_a, a.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, b.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));

    // Launch add() kernel on GPU.
    add<<<numOfBlocks, threadsPerBlock>>>(d_a, d_b, d_c, arraySize);
    // Check the kernel launch
    HIP_CHECK(hipGetLastError());
    // Check for kernel execution error
    HIP_CHECK(hipDeviceSynchronize());

    // Copy the result back to the host.
    HIP_CHECK(hipMemcpy(c.data(), d_c, arraySize * sizeof(int), hipMemcpyDeviceToHost));

    // Cleanup allocated memory.
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    // Print the result.
    std::cout << a[0] << " + " << b[0] << " = " << c[0] << std::endl;

    return EXIT_SUCCESS;
}
// [sphinx-end]
