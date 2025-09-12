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

int main()
{
    // [sphinx-start]
    double * ptr;
    HIP_CHECK(hipMalloc(&ptr, sizeof(double)));
    hipPointerAttribute_t attr;
    HIP_CHECK(hipPointerGetAttributes(&attr, ptr)); /*attr.type is hipMemoryTypeDevice*/
    if(attr.type == hipMemoryTypeDevice)
        std::cout << "ptr is of type hipMemoryTypeDevice" << std::endl;

    double* ptrHost;
    HIP_CHECK(hipHostMalloc(&ptrHost, sizeof(double)));
    hipPointerAttribute_t attrHost;
    HIP_CHECK(hipPointerGetAttributes(&attrHost, ptrHost)); /*attr.type is hipMemoryTypeHost*/
    if(attrHost.type == hipMemoryTypeHost)
        std::cout << "ptrHost is of type hipMemoryTypeHost" << std::endl;
    // [sphinx-end]
    
    HIP_CHECK(hipFreeHost(ptrHost));
    HIP_CHECK(hipFree(ptr));

    return EXIT_SUCCESS;
}
