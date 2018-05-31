/*
Copyright (c) 2015-Present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN: %t
 * HIT_END
 */

#include <hip/hip_runtime_api.h>
#include <iostream>
#include "test_common.h"

__global__
void fn(float* px, float* py)
{
    bool a[42];
    __shared__ double b[69];

    for (auto&& x : b) x = *py++;
    for (auto&& x : a) x = *px++ > 0.0;
    for (auto&& x : a) if (x) *--py = *--px;
}

int main() {
    hipInit(0);

    hipFuncAttributes attr{};

    auto r = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&fn));

    if (r != hipSuccess || attr.maxThreadsPerBlock == 0) {
        failed("Failed to read attributes.");
    }

    passed();
}
