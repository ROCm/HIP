// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

int main()
{
    // [sphinx-start]
    std::size_t elements = 1 << 20;
    std::size_t size_bytes = elements * sizeof(int);

    // allocate host and device memory
    int *host_pointer = new int[elements];
    int *device_input, *device_result;
    HIP_CHECK(hipMalloc(&device_input, size_bytes));
    HIP_CHECK(hipMalloc(&device_result, size_bytes));

    // copy from host to the device
    HIP_CHECK(hipMemcpy(device_input, host_pointer, size_bytes, hipMemcpyHostToDevice));

    // Use memory on the device, i.e. execute kernels

    // copy from device to host, to e.g. get results from the kernel
    HIP_CHECK(hipMemcpy(host_pointer, device_result, size_bytes, hipMemcpyDeviceToHost));

    // free memory when not needed any more
    HIP_CHECK(hipFree(device_result));
    HIP_CHECK(hipFree(device_input));
    delete[] host_pointer;
    // [sphinx-end]

    return EXIT_SUCCESS;
}
