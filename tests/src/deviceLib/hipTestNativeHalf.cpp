/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include <hip/hip_fp16.h>
#include "hip/hip_runtime.h"

#include "test_common.h"

#if __HIP_ARCH_GFX803__ || __HIP_ARCH_GFX900__ || __HIP_ARCH_GFX906__

__global__
__attribute__((optnone))
void __halfTest(bool* result) {
    // Construction
    __half a{1}; result[0] = __heq(a, 1);
    a = __half{1.0f}; result[0] = __heq(a, 1) && result[0];
    a = __half{1.0}; result[0] = __heq(a, 1) && result[0];
    a = __half{static_cast<unsigned short>(1)}; 
    result[0] = __heq(a, 1) && result[0];
    a = __half{static_cast<short>(1)}; result[0] = __heq(a, 1) && result[0];
    a = __half{1u}; result[0] = __heq(a, 1) && result[0];
    a = __half{1ul}; result[0] = __heq(a, 1) && result[0];
    a = __half{1l}; result[0] = __heq(a, 1) && result[0];
    a = __half{1ll}; result[0] = __heq(a, 1) && result[0];
    a = __half{1ull}; result[0] = __heq(a, 1) && result[0];

    // Assignment
    a = 0.0f; result[0] = __heq(a, 0) && result[0];
    a = 1.0; result[0] = __heq(a, 1) && result[0];
    a = __half_raw{2}; result[0] = __heq(a, 2) && result[0];

    // Nullary
    result[0] = __heq(a, +a) && result[0];
    result[0] = __heq(__hneg(a), -a) && result[0];

    // Unary arithmetic
    result[0] = __heq(a += 0, a) && result[0];
    result[0] = __heq(a -= 0, a) && result[0];
    result[0] = __heq(a *= 1, a) && result[0];
    result[0] = __heq(a /= 1, a) && result[0];

    // Binary arithmetic
    result[0] = __heq((a + a), __hadd(a, a)) && result[0];
    result[0] = __heq((a - a), __hsub(a, a)) && result[0];
    result[0] = __heq((a * a), __hmul(a, a)) && result[0];
    result[0] = __heq((a / a), __hdiv(a, a)) && result[0];

    // Relations
    result[0] = (a == a) && result[0];
    result[0] = !(a != a) && result[0];
    result[0] = (a <= a) && result[0];
    result[0] = (a >= a) && result[0];
    result[0] = !(a < a) && result[0];
    result[0] = !(a > a) && result[0];
}

__device__
bool to_bool(const __half2& x)
{
    auto r = static_cast<const __half2_raw&>(x);

    return r.data.x != 0 && r.data.y != 0;
}
__global__
__attribute__((optnone))
void __half2Test(bool* result) {
    // Construction
    __half2 a{1};
    result[0] = to_bool(__heq2(a, 1));
    a = __half2{__half{1}, __half{1}};
    result[0] = to_bool(__heq2(a, {1, 1})) && result[0];

    // Assignment
    a = __half2_raw{2}; result[0] = to_bool(__heq2(a, {2, 2})) && result[0];

    // Nullary
    result[0] = to_bool(__heq2(a, +a)) && result[0];
    result[0] = to_bool(__heq2(__hneg2(a), -a)) && result[0];

    // Unary arithmetic
    result[0] = to_bool(__heq2(a += 0, a)) && result[0];
    result[0] = to_bool(__heq2(a -= 0, a)) && result[0];
    result[0] = to_bool(__heq2(a *= 1, a)) && result[0];
    result[0] = to_bool(__heq2(a /= 1, a)) && result[0];

    // Binary arithmetic
    result[0] = to_bool(__heq2((a + a), __hadd2(a, a))) && result[0];
    result[0] = to_bool(__heq2((a - a), __hsub2(a, a))) && result[0];
    result[0] = to_bool(__heq2((a * a), __hmul2(a, a))) && result[0];
    result[0] = to_bool(__heq2((a / a), __h2div(a, a))) && result[0];

    // Relations
    result[0] = (a == a) && result[0];
    result[0] = !(a != a) && result[0];
    result[0] = (a <= a) && result[0];
    result[0] = (a >= a) && result[0];
    result[0] = !(a < a) && result[0];
    result[0] = !(a > a) && result[0];
}

#endif

int main() {
    bool* result{nullptr};
    hipHostMalloc(&result, 1);

    result[0] = false;
    hipLaunchKernelGGL(__halfTest, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result);
    hipDeviceSynchronize();

    if (!result[0]) { failed("Failed __half tests."); }

    result[0] = false;
    hipLaunchKernelGGL(__half2Test, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result);
    hipDeviceSynchronize();

    if (!result[0]) { failed("Failed __half2 tests."); }

    passed();
}
