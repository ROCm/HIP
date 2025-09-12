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

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if (err != hipSuccess)                                   \
    {                                                        \
        std::cout << "HIP Error: " << hipGetErrorString(err) \
              << " at line " << __LINE__ << std::endl;       \
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
}

__global__ void test_kernel()
{
    // [sphinx-start]
//#if __CUDA_ARCH__ >= 130 // does not properly specify, what feature is required, not portable
#if __HIP_ARCH_HAS_DOUBLES__ == 1 // explicitly specifies, what feature is required, portable between AMD and NVIDIA GPUs
    // device code
#endif
    // [sphinx-end]

#if __HIP_ARCH_HAS_DOUBLES__ == 1
    printf("Device has double-precision support.\n");
#else
    printf("Device does not have double-precision support.\n");
#endif
}

int main()
{
    test_kernel<<<1, 1, 0, 0>>>();
    HIP_CHECK(hipDeviceSynchronize());

    return EXIT_SUCCESS;
}
