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

#include <type_traits>

using namespace std;

#if __HIP_ARCH_GFX803__ || __HIP_ARCH_GFX900__ || __HIP_ARCH_GFX906__

__global__
void __halfTest(bool* result, __half a) {
    // Construction
    static_assert(is_default_constructible<__half>{}, "");
    static_assert(is_copy_constructible<__half>{}, "");
    static_assert(is_move_constructible<__half>{}, "");
    static_assert(is_constructible<__half, float>{}, "");
    static_assert(is_constructible<__half, double>{}, "");
    static_assert(is_constructible<__half, unsigned short>{}, "");
    static_assert(is_constructible<__half, short>{}, "");
    static_assert(is_constructible<__half, unsigned int>{}, "");
    static_assert(is_constructible<__half, int>{}, "");
    static_assert(is_constructible<__half, unsigned long>{}, "");
    static_assert(is_constructible<__half, long>{}, "");
    static_assert(is_constructible<__half, long long>{}, "");
    static_assert(is_constructible<__half, unsigned long long>{}, "");
    static_assert(is_constructible<__half, __half_raw>{}, "");

    // Assignment
    static_assert(is_copy_assignable<__half>{}, "");
    static_assert(is_move_assignable<__half>{}, "");
    static_assert(is_assignable<__half, float>{}, "");
    static_assert(is_assignable<__half, double>{}, "");
    static_assert(is_assignable<__half, unsigned short>{}, "");
    static_assert(is_assignable<__half, short>{}, "");
    static_assert(is_assignable<__half, unsigned int>{}, "");
    static_assert(is_assignable<__half, int>{}, "");
    static_assert(is_assignable<__half, unsigned long>{}, "");
    static_assert(is_assignable<__half, long>{}, "");
    static_assert(is_assignable<__half, long long>{}, "");
    static_assert(is_assignable<__half, unsigned long long>{}, "");
    static_assert(is_assignable<__half, __half_raw>{}, "");
    static_assert(is_assignable<__half, volatile __half_raw&>{}, "");
    static_assert(is_assignable<__half, volatile __half_raw&&>{}, "");

    // Conversion
    static_assert(is_convertible<__half, float>{}, "");
    static_assert(is_convertible<__half, unsigned short>{}, "");
    static_assert(is_convertible<__half, short>{}, "");
    static_assert(is_convertible<__half, unsigned int>{}, "");
    static_assert(is_convertible<__half, int>{}, "");
    static_assert(is_convertible<__half, unsigned long>{}, "");
    static_assert(is_convertible<__half, long>{}, "");
    static_assert(is_convertible<__half, long long>{}, "");
    static_assert(is_convertible<__half, bool>{}, "");
    static_assert(is_convertible<__half, unsigned long long>{}, "");
    static_assert(is_convertible<__half, __half_raw>{}, "");
    static_assert(is_convertible<__half, volatile __half_raw>{}, "");

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
void __half2Test(bool* result, __half2 a) {
    // Construction
    static_assert(is_default_constructible<__half2>{}, "");
    static_assert(is_copy_constructible<__half2>{}, "");
    static_assert(is_move_constructible<__half2>{}, "");
    static_assert(is_constructible<__half2, __half, __half>{}, "");
    static_assert(is_constructible<__half2, __half2_raw>{}, "");

    // Assignment
    static_assert(is_copy_assignable<__half2>{}, "");
    static_assert(is_move_assignable<__half2>{}, "");
    static_assert(is_assignable<__half2, __half2_raw>{}, "");

    // Conversion
    static_assert(is_convertible<__half2, __half2_raw>{}, "");

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

    result[0] = true;
    hipLaunchKernelGGL(
        __halfTest, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half{1});
    hipDeviceSynchronize();

    if (!result[0]) { failed("Failed __half tests."); }

    result[0] = true;
    hipLaunchKernelGGL(
        __half2Test, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half2{1, 1});
    hipDeviceSynchronize();

    if (!result[0]) { failed("Failed __half2 tests."); }

    hipHostFree(result);

    passed();
}
