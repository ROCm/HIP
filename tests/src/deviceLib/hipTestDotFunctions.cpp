/*
Copyright (c) 2015-2019 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t EXCLUDE_HIP_PLATFORM nvcc
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <hip/math_functions.h>
#include "test_common.h"


__global__ void DotFunctions(bool* result) {
    #if (__hcc_workweek__ >= 19015) || __HIP_CLANG_ONLY__
    // Dot Functions
    short2 sa{1}, sb{1};
    result[0] = amd_mixed_dot(sa, sb, 1, result[0]) && result[0];

    ushort2 usa{1}, usb{1};
    result[0] = amd_mixed_dot(usa, usb, (uint) 1, result[0]) && result[0];

    char4 ca{1}, cb{1};
    result[0] = amd_mixed_dot(ca, cb, 1, result[0]) && result[0];

    uchar4 uca{1}, ucb{1};
    result[0] = amd_mixed_dot(uca, ucb, (uint) 1, result[0]) && result[0];

    int ia{1}, ib{1};
    result[0] = amd_mixed_dot(ia, ib, 1, result[0]) && result[0];

    uint ua{1}, ub{1};
    result[0] = amd_mixed_dot(ua, ub, (uint) 1, result[0]) && result[0];
    #endif
}

int main() {
    bool* result{nullptr};
    hipHostMalloc(&result, 1);
    result[0] = true;

    hipLaunchKernelGGL(DotFunctions, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result);
    hipDeviceSynchronize();
    if (!result[0]) { failed("Failed dot tests."); }

    hipHostFree(result);

    passed();
}
