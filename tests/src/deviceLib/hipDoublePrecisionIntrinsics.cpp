/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include "test_common.h"

#pragma GCC diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wunused-variable"

__device__ void double_precision_intrinsics() {
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __dadd_rd(0.0, 1.0);
#endif
    __dadd_rn(0.0, 1.0);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __dadd_ru(0.0, 1.0);
    __dadd_rz(0.0, 1.0);
    __ddiv_rd(0.0, 1.0);
#endif
    __ddiv_rn(0.0, 1.0);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __ddiv_ru(0.0, 1.0);
    __ddiv_rz(0.0, 1.0);
    __dmul_rd(1.0, 2.0);
#endif
    __dmul_rn(1.0, 2.0);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __dmul_ru(1.0, 2.0);
    __dmul_rz(1.0, 2.0);
    __drcp_rd(2.0);
#endif
    __drcp_rn(2.0);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __drcp_ru(2.0);
    __drcp_rz(2.0);
    __dsqrt_rd(4.0);
#endif
    __dsqrt_rn(4.0);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __dsqrt_ru(4.0);
    __dsqrt_rz(4.0);
    __dsub_rd(2.0, 1.0);
#endif
    __dsub_rn(2.0, 1.0);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __dsub_ru(2.0, 1.0);
    __dsub_rz(2.0, 1.0);
    __fma_rd(1.0, 2.0, 3.0);
#endif
    __fma_rn(1.0, 2.0, 3.0);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
    __fma_ru(1.0, 2.0, 3.0);
    __fma_rz(1.0, 2.0, 3.0);
#endif
}

__global__ void compileDoublePrecisionIntrinsics(int ignored) {
    double_precision_intrinsics();
}

int main() {
    hipLaunchKernelGGL(compileDoublePrecisionIntrinsics, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, 1);
    passed();
}
