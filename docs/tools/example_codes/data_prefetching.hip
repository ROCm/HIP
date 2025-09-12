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
    int *a, *b, *c;
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId)); // Get the current device ID

    // Allocate memory for a, b and c that is accessible to both device and host codes.
    HIP_CHECK(hipMallocManaged(&a, sizeof(*a)));
    HIP_CHECK(hipMallocManaged(&b, sizeof(*b)));
    HIP_CHECK(hipMallocManaged(&c, sizeof(*c)));

    // Setup input values.
    *a = 1;
    *b = 2;

    // Prefetch the data to the GPU device.
    HIP_CHECK(hipMemPrefetchAsync(a, sizeof(*a), deviceId, 0));
    HIP_CHECK(hipMemPrefetchAsync(b, sizeof(*b), deviceId, 0));
    HIP_CHECK(hipMemPrefetchAsync(c, sizeof(*c), deviceId, 0));

    // Launch add() kernel on GPU.
    add<<<1, 1>>>(a, b, c);

    // Prefetch the result back to the CPU.
    HIP_CHECK(hipMemPrefetchAsync(c, sizeof(*c), hipCpuDeviceId, 0));

    // Wait for the prefetch operations to complete.
    HIP_CHECK(hipDeviceSynchronize());

    // Prints the result.
    std::cout << *a << " + " << *b << " = " << *c << std::endl;

    // Cleanup allocated memory.
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(c));

    return 0;
}
// [sphinx-end]