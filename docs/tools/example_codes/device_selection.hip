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

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

#define HIP_CHECK(expression)                        \
{                                                    \
    const hipError_t status = expression;            \
    if (status != hipSuccess)                        \
    {                                                \
        std::cerr << "HIP error " << status          \
                << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":"         \
                << __LINE__ << std::endl;            \
        std::exit(EXIT_FAILURE);                     \
    }                                                \
}

__global__ void simpleKernel(double *data, std::size_t elems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < elems)
        data[idx] = idx * 2.0;
}

int main()
{
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if(deviceCount < 2)
    {
        std::cout << "This example requires at least two HIP devices." << std::endl;
        return EXIT_SUCCESS;
    }

    double* deviceData0;
    double* deviceData1;
    constexpr std::size_t elems = 1024;
    constexpr std::size_t size = elems * sizeof(double);

    int deviceId0 = 0;
    int deviceId1 = 1;

    // Set device 0 and perform operations
    HIP_CHECK(hipSetDevice(deviceId0)); // Set device 0 as current
    HIP_CHECK(hipMalloc(&deviceData0, size)); // Allocate memory on device 0
    simpleKernel<<<8, 128>>>(deviceData0, elems); // Launch kernel on device 0
    HIP_CHECK(hipDeviceSynchronize());

    // Set device 1 and perform operations
    HIP_CHECK(hipSetDevice(deviceId1)); // Set device 1 as current
    HIP_CHECK(hipMalloc(&deviceData1, size)); // Allocate memory on device 1
    simpleKernel<<<8, 128>>>(deviceData1, elems); // Launch kernel on device 1
    HIP_CHECK(hipDeviceSynchronize());

    // Copy result from device 0
    double hostData0[elems];
    HIP_CHECK(hipSetDevice(deviceId0));
    HIP_CHECK(hipMemcpy(hostData0, deviceData0, size, hipMemcpyDeviceToHost));

    // Copy result from device 1
    double hostData1[elems];
    HIP_CHECK(hipSetDevice(deviceId1));
    HIP_CHECK(hipMemcpy(hostData1, deviceData1, size, hipMemcpyDeviceToHost));

    // Display results from both devices
    std::cout << "Device 0 data: " << hostData0[0] << std::endl;
    std::cout << "Device 1 data: " << hostData1[0] << std::endl;

    // Free device memory
    HIP_CHECK(hipFree(deviceData0));
    HIP_CHECK(hipFree(deviceData1));

    return EXIT_SUCCESS;
}
// [sphinx-end]
