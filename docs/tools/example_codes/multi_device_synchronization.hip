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
    int numDevices;
    HIP_CHECK(hipGetDeviceCount(&numDevices));

    if (numDevices < 2)
    {
        std::cout << "This example requires at least two HIP devices." << std::endl;
        return EXIT_SUCCESS;
    }

    double *deviceData0;
    double *deviceData1;
    constexpr std::size_t elems = 1024;
    constexpr std::size_t size = elems * sizeof(double);

    // Create streams and events for each device
    hipStream_t stream0, stream1;
    hipEvent_t startEvent0, stopEvent0, startEvent1, stopEvent1;

    // Initialize device 0
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamCreate(&stream0));
    HIP_CHECK(hipEventCreate(&startEvent0));
    HIP_CHECK(hipEventCreate(&stopEvent0));
    HIP_CHECK(hipMalloc(&deviceData0, size));

    // Initialize device 1
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipEventCreate(&startEvent1));
    HIP_CHECK(hipEventCreate(&stopEvent1));
    HIP_CHECK(hipMalloc(&deviceData1, size));

    // Record the start event on device 0
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventRecord(startEvent0, stream0));

    // Launch the kernel asynchronously on device 0
    simpleKernel<<<8, 128, 0, stream0>>>(deviceData0, elems);

    // Record the stop event on device 0
    HIP_CHECK(hipEventRecord(stopEvent0, stream0));

    // Wait for the stop event on device 0 to complete
    HIP_CHECK(hipEventSynchronize(stopEvent0));

    // Record the start event on device 1
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipEventRecord(startEvent1, stream1));

    // Launch the kernel asynchronously on device 1
    simpleKernel<<<8, 128, 0, stream1>>>(deviceData1, elems);

    // Record the stop event on device 1
    HIP_CHECK(hipEventRecord(stopEvent1, stream1));

    // Wait for the stop event on device 1 to complete
    HIP_CHECK(hipEventSynchronize(stopEvent1));

    // Calculate elapsed time between the events for both devices
    float milliseconds0 = 0, milliseconds1 = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds0, startEvent0, stopEvent0));
    HIP_CHECK(hipEventElapsedTime(&milliseconds1, startEvent1, stopEvent1));

    std::cout << "Elapsed time on GPU 0: " << milliseconds0 << " ms" << std::endl;
    std::cout << "Elapsed time on GPU 1: " << milliseconds1 << " ms" << std::endl;

    // Cleanup for device 0
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventDestroy(startEvent0));
    HIP_CHECK(hipEventDestroy(stopEvent0));
    HIP_CHECK(hipStreamSynchronize(stream0));
    HIP_CHECK(hipStreamDestroy(stream0));
    HIP_CHECK(hipFree(deviceData0));

    // Cleanup for device 1
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipEventDestroy(startEvent1));
    HIP_CHECK(hipEventDestroy(stopEvent1));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipFree(deviceData1));

    return EXIT_SUCCESS;
}
// [sphinx-end]
