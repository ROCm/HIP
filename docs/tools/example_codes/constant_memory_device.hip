// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
