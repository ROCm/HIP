// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

// [sphinx-start]
constexpr std::size_t const_array_size = 32;
__constant__ double const_array[const_array_size];

void set_constant_memory(double* values)
{
    HIP_CHECK(hipMemcpyToSymbol(const_array, values, const_array_size * sizeof(double)));
}

__global__ void kernel_using_const_memory(double* array)
{
    int warpIdx = threadIdx.x / warpSize;
    // uniform access of warps to const_array for best performance
    array[blockIdx.x] *= const_array[warpIdx];
}
// [sphinx-end]

int main()
{
    std::size_t elements = 32;
    std::size_t size_bytes = elements * sizeof(double);

    // allocate host array
    double *host_array = new double[elements];

    // allocate device array
    double *device_array = nullptr;
    HIP_CHECK(hipMalloc((double**) &device_array, size_bytes));

    // copy from host to the device
    set_constant_memory(host_array);

    kernel_using_const_memory<<<32, 32>>>(device_array);

    // copy from device to host, to e.g. get results from the kernel
    HIP_CHECK(hipMemcpy(host_array, device_array, size_bytes, hipMemcpyDeviceToHost));

    // free memory when not needed any more
    HIP_CHECK(hipFree(device_array));
    delete[] host_array;

    std::cout << "Success!" << std::endl;

    return EXIT_SUCCESS;
}
