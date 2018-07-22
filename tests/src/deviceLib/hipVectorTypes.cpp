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

#include <hip/hip_vector_types.h>

#include "vector_test_common.h"
#include "test_common.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <type_traits>

using namespace std;

template<
    typename V,
    Enable_if_t<!is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_unary_tests(V&, V&) {
    return true;
}

template<
    typename V,
    Enable_if_t<!is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_binary_tests(V&, V&, V&...) {
    return true;
}

template<
    typename V,
    Enable_if_t<is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_unary_tests(V& f1, V& f2) {
    f1 %= f2;
    if (f1 != V{0}) return false;
    f1 &= f2;
    if (f1 != V{0}) return false;
    f1 |= f2;
    if (f1 != V{1}) return false;
    f1 ^= f2;
    if (f1 != V{0}) return false;
    f1 = V{1};
    f1 <<= f2;
    if (f1 != V{2}) return false;
    f1 >>= f2;
    if (f1 != V{1}) return false;
    f2 = ~f1;
    return f2 == V{~1};
}

template<
    typename V,
    Enable_if_t<is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_binary_tests(V& f1, V& f2, V& f3) {
    f3 = f1 % f2;
    if (f3 != V{0}) return false;
    f1 = f3 & f2;
    if (f1 != V{0}) return false;
    f2 = f1 ^ f3;
    if (f2 != V{0}) return false;
    f1 = V{1};
    f2 = V{2};
    f3 = f1 << f2;
    if (f3 != V{4}) return false;
    f2 = f3 >> f1;
    return f2 == V{2};
}

template<typename V>
bool constructor_tests() {
    static_assert(is_constructible<V, unsigned char>{}, "");
    static_assert(is_constructible<V, signed char>{}, "");
    static_assert(is_constructible<V, unsigned short>{}, "");
    static_assert(is_constructible<V, signed short>{}, "");
    static_assert(is_constructible<V, unsigned int>{}, "");
    static_assert(is_constructible<V, signed int>{}, "");
    static_assert(is_constructible<V, unsigned long>{}, "");
    static_assert(is_constructible<V, signed long>{}, "");
    static_assert(is_constructible<V, unsigned long long>{}, "");
    static_assert(is_constructible<V, signed long long>{}, "");
    static_assert(is_constructible<V, float>{}, "");
    static_assert(is_constructible<V, double>{}, "");

    return true;
}

template<typename V>
bool TestVectorType() {
    V f1{1};
    V f2{1};
    V f3 = f1 + f2;
    if (f3 != V{2}) return false;
    f2 = f3 - f1;
    if (f2 != V{1}) return false;
    f1 = f2 * f3;
    if (f1 != V{2}) return false;
    f2 = f1 / f3;
    if (f2 != V{1}) return false;
    if (!integer_binary_tests(f1, f2, f3)) return false;

    f1 = V{2};
    f2 = V{1};
    f1 += f2;
    if (f1 != V{3}) return false;
    f1 -= f2;
    if (f1 != V{2}) return false;
    f1 *= f2;
    if (f1 != V{2}) return false;
    f1 /= f2;
    if (f1 != V{2}) return false;
    if (!integer_unary_tests(f1, f2)) return false;

    f1 = V{2};
    f2 = f1++;
    if (f1 != V{3}) return false;
    if (f2 != V{2}) return false;
    f2 = f1--;
    if (f2 != V{3}) return false;
    if (f1 != V{2}) return false;
    f2 = ++f1;
    if (f1 != V{3}) return false;
    if (f2 != V{3}) return false;
    f2 = --f1;
    if (f1 != V{2}) return false;
    if (f2 != V{2}) return false;

    if (!constructor_tests<V>()) return false;

    f1 = V{3};
    f2 = V{4};
    f3 = V{3};
    if (f1 == f2) return false;
    if (!(f1 != f2)) return false;

    return true;
}

template<typename... Ts, Enable_if_t<sizeof...(Ts) == 0>* = nullptr>
bool TestVectorTypes() {
    return true;
}

template<typename T, typename... Ts>
bool TestVectorTypes() {
    if (!TestVectorType<T>()) return false;
    return TestVectorTypes<Ts...>();
}

bool CheckVectorTypes() {
    return TestVectorTypes<
        char1, char2, char3, char4,
        uchar1, uchar2, uchar3, uchar4,
        short1, short2, short3, short4,
        ushort1, ushort2, ushort3, ushort4,
        int1, int2, int3, int4,
        uint1, uint2, uint3, uint4,
        long1, long2, long3, long4,
        ulong1, ulong2, ulong3, ulong4,
        longlong1, longlong2, longlong3, longlong4,
        ulonglong1, ulonglong2, ulonglong3, ulonglong4,
        float1, float2, float3, float4,
        double1, double2, double3, double4>();
}

int main() {
    static_assert(sizeof(float1) == 4, "");
    static_assert(sizeof(float2) >= 8, "");
    static_assert(sizeof(float3) >= 12, "");
    static_assert(sizeof(float4) >= 16, "");

    if (CheckVectorTypes()) {
        float1 f1 = make_float1(1.0f);
        passed();
    }
    else {
        failed("Failed some vector test on the host side.");
    }
}