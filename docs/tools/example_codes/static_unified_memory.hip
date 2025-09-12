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
