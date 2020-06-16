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
 * TEST: %t
 * HIT_END
 */

#include <hip/hip_vector_types.h>

#include "vector_test_common.h"
#include "test_common.h"

#include <memory>
#include <type_traits>
#include <utility>

using namespace std;

template <class T> __device__
typename std::add_rvalue_reference<T>::type _declval() noexcept;

template<
    typename V,
    Enable_if_t<!is_integral<decltype(_declval<V>().x)>{}>* = nullptr>
__device__
constexpr
bool integer_unary_tests(const V&, const V&) {
    return true;
}

template<
    typename V,
    Enable_if_t<is_integral<decltype(_declval<V>().x)>{}>* = nullptr>
__device__
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

    return true;
}

template<
    typename V,
    Enable_if_t<!is_integral<decltype(_declval<V>().x)>{}>* = nullptr>
__device__
constexpr
bool integer_binary_tests(const V&, const V&, const V&) {
    return true;
}

template<
    typename V,
    Enable_if_t<is_integral<decltype(_declval<V>().x)>{}>* = nullptr>
__device__
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
__device__
bool TestVectorType() {
    constexpr V v1{1};
    constexpr V v2{2};
    constexpr V v3{3};
    constexpr V v4{4};

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

    f1 = v2;
    f2 = v1;
    f1 += f2;
    if (f1 != v3) return false;
    f1 -= f2;
    if (f1 != v2) return false;
    f1 *= f2;
    if (f1 != v2) return false;
    f1 /= f2;
    if (f1 != v2) return false;
    if (!integer_unary_tests(f1, f2)) return false;

    f1 = v2;
    f2 = f1++;
    if (f1 != v3) return false;
    if (f2 != v2) return false;
    f2 = f1--;
    if (f2 != v3) return false;
    if (f1 != v2) return false;
    f2 = ++f1;
    if (f1 != v3) return false;
    if (f2 != v3) return false;
    f2 = --f1;
    if (f1 != v2) return false;
    if (f2 != v2) return false;

    f1 = v3;
    f2 = v4;
    f3 = v3;
    if (f1 == f2) return false;
    if (!(f1 != f2)) return false;

    #if 0 // TODO: investigate on GFX8
        using T = typename V::value_type;

        const T& x = f1.x;
        T& y = f2.x;
        const volatile T& z = f3.x;
        volatile T& w = f2.x;

        if (x != T{3}) return false;
        if (y != T{4}) return false;
        if (z != T{3}) return false;
        if (w != T{4}) return false;
    #endif

    return true;
}

template<typename... Ts, Enable_if_t<sizeof...(Ts) == 0>* = nullptr>
__device__
bool TestVectorTypes() {
    return true;
}

template<typename T, typename... Ts>
__device__
bool TestVectorTypes() {
    if (!TestVectorType<T>()) return false;
    return TestVectorTypes<Ts...>();
}

__global__
void CheckVectorTypes(bool* ptr) {
    ptr[0] = TestVectorTypes<
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


template<typename V>
__global__
void CheckSharedVectorType(bool* ptr) {
    constexpr V v1{1};
    constexpr V v2{2};
    constexpr V v3{3};
    constexpr V v4{4};
    __shared__ V f1, f2, f3;

    *ptr = true;
    f1 = V{1};
    f2 = V{1};
    f3 = f1 + f2;
    *ptr = *ptr && f3 == V{2};
    f2 = f3 - f1;
    *ptr = *ptr && f2 == V{1};
    f1 = f2 * f3;
    *ptr = *ptr && f1 == V{2};
    f2 = f1 / f3;
    *ptr = *ptr && f2 == V{1};
    *ptr = *ptr && integer_binary_tests(f1, f2, f3);

    f1 = v2;
    f2 = v1;
    f1 += f2;
    *ptr = *ptr && f1 == v3;
    f1 -= f2;
    *ptr = *ptr && f1 == v2;
    f1 *= f2;
    *ptr = *ptr && f1 == v2;
    f1 /= f2;
    *ptr = *ptr && f1 == v2;
    *ptr = *ptr && integer_unary_tests(f1, f2);

    f1 = v2;
    f2 = f1++;
    *ptr = *ptr && f1 == v3;
    *ptr = *ptr && f2 == v2;
    f2 = f1--;
    *ptr = *ptr && f2 == v3;
    *ptr = *ptr && f1 == v2;
    f2 = ++f1;
    *ptr = *ptr && f1 == v3;
    *ptr = *ptr && f2 == v3;
    f2 = --f1;
    *ptr = *ptr && f1 == v2;
    *ptr = *ptr && f2 == v2;

    f1 = v3;
    f2 = v4;
    f3 = v3;
    *ptr = *ptr && f1 != f2;
}

template <typename V>
bool run_CheckSharedVectorType() {
    bool* ptr = nullptr;
    if (hipMalloc(&ptr, sizeof(bool)) != HIP_SUCCESS) return false;
    unique_ptr<bool, decltype(hipFree)*> correct{ptr, hipFree};
    hipLaunchKernelGGL(
        (CheckSharedVectorType<V>), dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, correct.get());
    bool passed = true;
    if (hipMemcpyDtoH(&passed, correct.get(), sizeof(bool)) != HIP_SUCCESS) {
        return false;
    }
    return passed;
}

template<typename... Ts, Enable_if_t<sizeof...(Ts) == 0>* = nullptr>
bool run_CheckSharedVectorTypes() {
    return true;
}

template <typename V, typename... Vs>
bool run_CheckSharedVectorTypes() {
    return run_CheckSharedVectorType<V>() &&
             run_CheckSharedVectorTypes<Vs...>();
}

int main() {
    static_assert(sizeof(float1) == 4, "");
    static_assert(sizeof(float2) >= 8, "");
    static_assert(sizeof(float3) >= 12, "");
    static_assert(sizeof(float4) >= 16, "");

    bool* ptr = nullptr;
    if (hipMalloc(&ptr, sizeof(bool)) != HIP_SUCCESS) return EXIT_FAILURE;
    unique_ptr<bool, decltype(hipFree)*> correct{ptr, hipFree};
    hipLaunchKernelGGL(
        CheckVectorTypes, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, correct.get());
    bool passed = true;
    if (hipMemcpyDtoH(&passed, correct.get(), sizeof(bool)) != HIP_SUCCESS) {
        return EXIT_FAILURE;
    }

    passed = passed && run_CheckSharedVectorTypes<
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

    if (passed == true) {
        passed();
    }
    else {
        failed("Failed some vector test.");
    }
}
