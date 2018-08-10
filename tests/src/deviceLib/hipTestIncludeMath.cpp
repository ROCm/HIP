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

// Test include math_functions.h then hip_runtime.h.
// Incorrect implementation causes compilation failure due to conflict
// declartions.

#include <hip/math_functions.h>

// Test __HIP_DEVICE_COMPILE__ is defined after math_functions.h
// is included.
//
__device__ __host__ inline void throw_std_bad_alloc()
{
  #ifndef __HIP_DEVICE_COMPILE__
    throw std::bad_alloc();
  #else
    std::size_t huge = static_cast<std::size_t>(-1);
    new int[huge];
  #endif
}

// Test __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ and __HIP_ARCH_HAS_DYNAMIC_PARALLEL__
// is defined. Eigen HIP/hcc/Half.h __ldg depends on this.
#if !defined(__HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__) ||                            \
    !defined(__HIP_ARCH_HAS_DYNAMIC_PARALLEL__)
#error                                                                         \
    "__HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ or __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ not defined"
#endif

#include <hip/hip_runtime.h>
#include "test_common.h"

__global__ void FloatMathPrecise() {
    int iX;
    float fX, fY;

    acosf(1.0f);
    acoshf(1.0f);
    asinf(0.0f);
    asinhf(0.0f);
    atan2f(0.0f, 1.0f);
    atanf(0.0f);
    atanhf(0.0f);
    cbrtf(0.0f);
    fX = ceilf(0.0f);
    fX = copysignf(1.0f, -2.0f);
    cosf(0.0f);
    coshf(0.0f);
    cospif(0.0f);
    cyl_bessel_i0f(0.0f);
    cyl_bessel_i1f(0.0f);
    erfcf(0.0f);
    erfcinvf(2.0f);
    erfcxf(0.0f);
    erff(0.0f);
    erfinvf(1.0f);
    exp10f(0.0f);
    exp2f(0.0f);
    expf(0.0f);
    expm1f(0.0f);
    fX = fabsf(1.0f);
    fdimf(1.0f, 0.0f);
    fdividef(0.0f, 1.0f);
    fX = floorf(0.0f);
    fmaf(1.0f, 2.0f, 3.0f);
    fX = fmaxf(0.0f, 0.0f);
    fX = fminf(0.0f, 0.0f);
    fmodf(0.0f, 1.0f);
    frexpf(0.0f, &iX);
    hypotf(1.0f, 0.0f);
    ilogbf(1.0f);
    isfinite(0.0f);
    fX = isinf(0.0f);
    fX = isnan(0.0f);
    j0f(0.0f);
    j1f(0.0f);
    jnf(-1.0f, 1.0f);
    ldexpf(0.0f, 0);
    lgammaf(1.0f);
    llrintf(0.0f);
    llroundf(0.0f);
    log10f(1.0f);
    log1pf(-1.0f);
    log2f(1.0f);
    logbf(1.0f);
    logf(1.0f);
    lrintf(0.0f);
    lroundf(0.0f);
    modff(0.0f, &fX);
    fX = nanf("1");
    fX = nearbyintf(0.0f);
    nextafterf(0.0f, 0.0f);
    norm3df(1.0f, 0.0f, 0.0f);
    norm4df(1.0f, 0.0f, 0.0f, 0.0f);
    normcdff(0.0f);
    normcdfinvf(1.0f);
    fX = 1.0f;
    normf(1, &fX);
    powf(1.0f, 0.0f);
    rcbrtf(1.0f);
    remainderf(2.0f, 1.0f);
    remquof(1.0f, 2.0f, &iX);
    rhypotf(0.0f, 1.0f);
    fY = rintf(1.0f);
    rnorm3df(0.0f, 0.0f, 1.0f);
    rnorm4df(0.0f, 0.0f, 0.0f, 1.0f);
    fX = 1.0f;
    rnormf(1, &fX);
    fY = roundf(0.0f);
    rsqrtf(1.0f);
    scalblnf(0.0f, 1);
    scalbnf(0.0f, 1);
    signbit(1.0f);
    sincosf(0.0f, &fX, &fY);
    sincospif(0.0f, &fX, &fY);
    sinf(0.0f);
    sinhf(0.0f);
    sinpif(0.0f);
    sqrtf(0.0f);
    tanf(0.0f);
    tanhf(0.0f);
    tgammaf(2.0f);
    fY = truncf(0.0f);
    y0f(1.0f);
    y1f(1.0f);
    ynf(1, 1.0f);
}

int main() {
    hipLaunchKernelGGL(FloatMathPrecise, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0);
    passed();
}
