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

__device__ void single_precision_intrinsics()
{
    float fX, fY;

    __cosf(0.0f);
    __exp10f(0.0f);
    __expf(0.0f);
    __fadd_rd(0.0f, 1.0f);
    __fadd_rn(0.0f, 1.0f);
    __fadd_ru(0.0f, 1.0f);
    __fadd_rz(0.0f, 1.0f);
    __fdiv_rd(4.0f, 2.0f);
    __fdiv_rn(4.0f, 2.0f);
    __fdiv_ru(4.0f, 2.0f);
    __fdiv_rz(4.0f, 2.0f);
    __fdividef(4.0f, 2.0f);
    __fmaf_rd(1.0f, 2.0f, 3.0f);
    __fmaf_rn(1.0f, 2.0f, 3.0f);
    __fmaf_ru(1.0f, 2.0f, 3.0f);
    __fmaf_rz(1.0f, 2.0f, 3.0f);
    __fmul_rd(1.0f, 2.0f);
    __fmul_rn(1.0f, 2.0f);
    __fmul_ru(1.0f, 2.0f);
    __fmul_rz(1.0f, 2.0f);
    __frcp_rd(2.0f);
    __frcp_rn(2.0f);
    __frcp_ru(2.0f);
    __frcp_rz(2.0f);
    __frsqrt_rn(4.0f);
    __fsqrt_rd(4.0f);
    __fsqrt_rn(4.0f);
    __fsqrt_ru(4.0f);
    __fsqrt_rz(4.0f);
    __fsub_rd(2.0f, 1.0f);
    __fsub_rn(2.0f, 1.0f);
    __fsub_ru(2.0f, 1.0f);
    __fsub_rz(2.0f, 1.0f);
    __log10f(1.0f);
    __log2f(1.0f);
    __logf(1.0f);
    __powf(1.0f, 0.0f);
    __saturatef(0.1f);
    __sincosf(0.0f, &fX, &fY);
    __sinf(0.0f);
    __tanf(0.0f);
}


__global__ void compileSinglePrecisionIntrinsics(hipLaunchParm lp, int ignored)
{
    single_precision_intrinsics();
}


int main()
{
  hipLaunchKernel(compileSinglePrecisionIntrinsics, dim3(1,1,1), dim3(1,1,1), 0, 0, 1);
  passed();
}
