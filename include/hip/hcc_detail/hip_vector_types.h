/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

/**
 *  @file  hcc_detail/hip_vector_types.h
 *  @brief Defines the different newt vector types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H

#if defined(__HCC__) && (__hcc_workweek__ < 16032)
#error("This version of HIP requires a newer version of HCC.");
#endif

#include "hip/hcc_detail/host_defines.h"

#include <type_traits>

#if defined(__clang__)
    #define __NATIVE_VECTOR__(n, ...) __attribute__((ext_vector_type(n)))
#elif defined(__GNUC__) // N.B.: GCC does not support .xyzw syntax.
    #define __ROUND_UP_TO_NEXT_POT__(x) \
        (1 << (31 - __builtin_clz(x) + (x > (1 << (31 - __builtin_clz(x))))))
    #define __NATIVE_VECTOR__(n, T) \
        __attribute__((vector_size(__ROUND_UP_TO_NEXT_POT__(n) * sizeof(T))))
#endif

#if defined(__cplusplus)
    template<typename T, unsigned int n> struct HIP_vector_base;

    template<typename T>
    struct HIP_vector_base<T, 1> {
        typedef T Native_vec_ __NATIVE_VECTOR__(1, T);

        union {
            Native_vec_ data;
            struct {
                typename std::decay<
                    decltype(std::declval<Native_vec_>().x)>::type x;
            };
        };
    };

    template<typename T>
    struct HIP_vector_base<T, 2> {
        typedef T Native_vec_ __NATIVE_VECTOR__(2, T);

        union {
            Native_vec_ data;
            struct {
                typename std::decay<
                    decltype(std::declval<Native_vec_>().x)>::type x;
                typename std::decay<
                    decltype(std::declval<Native_vec_>().y)>::type y;
            };
        };
    };

    template<typename T>
    struct HIP_vector_base<T, 3> {
        typedef T Native_vec_ __NATIVE_VECTOR__(3, T);

        union {
            Native_vec_ data;
            struct {
                typename std::decay<
                    decltype(std::declval<Native_vec_>().x)>::type x;
                typename std::decay<
                    decltype(std::declval<Native_vec_>().y)>::type y;
                typename std::decay<
                    decltype(std::declval<Native_vec_>().z)>::type z;
            };
        };
    };

    template<typename T>
    struct HIP_vector_base<T, 4> {
        typedef T Native_vec_ __NATIVE_VECTOR__(4, T);

        union {
            Native_vec_ data;
            struct {
                typename std::decay<
                    decltype(std::declval<Native_vec_>().x)>::type x;
                typename std::decay<
                    decltype(std::declval<Native_vec_>().y)>::type y;
                typename std::decay<
                    decltype(std::declval<Native_vec_>().z)>::type z;
                typename std::decay<
                    decltype(std::declval<Native_vec_>().w)>::type w;
            };
        };
    };

    template<typename T, unsigned int rank>
    struct HIP_vector_type : public HIP_vector_base<T, rank> {
        using HIP_vector_base<T, rank>::data;
        using typename HIP_vector_base<T, rank>::Native_vec_;

        __host__ __device__
        HIP_vector_type() = default;
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __host__ __device__
        explicit
        HIP_vector_type(U x) noexcept { data = Native_vec_(x); }
        template< // TODO: constrain based on type as well.
            typename... Us,
            typename std::enable_if<sizeof...(Us) == rank>::type* = nullptr>
        __host__ __device__
        HIP_vector_type(Us... xs) noexcept { data = Native_vec_{xs...}; }
        __host__ __device__
        HIP_vector_type(const HIP_vector_type&) = default;
        __host__ __device__
        HIP_vector_type(HIP_vector_type&&) = default;
        __host__ __device__
        ~HIP_vector_type() = default;

        __host__ __device__
        HIP_vector_type& operator=(const HIP_vector_type&) = default;
        __host__ __device__
        HIP_vector_type& operator=(HIP_vector_type&&) = default;

        // Operators
        __host__ __device__
        HIP_vector_type& operator++() noexcept
        {
            data += Native_vec_(1);
            return *this;
        }
        __host__ __device__
        HIP_vector_type operator++(int) noexcept
        {
            auto tmp(*this);
            ++*this;
            return tmp;
        }
        __host__ __device__
        HIP_vector_type& operator--() noexcept
        {
            data -= Native_vec_(1);
            return *this;
        }
        __host__ __device__
        HIP_vector_type operator--(int) noexcept
        {
            auto tmp(*this);
            --*this;
            return tmp;
        }
        __host__ __device__
        HIP_vector_type& operator+=(const HIP_vector_type& x) noexcept
        {
            data += x.data;
            return *this;
        }
        __host__ __device__
        HIP_vector_type& operator-=(const HIP_vector_type& x) noexcept
        {
            data -= x.data;
            return *this;
        }
        __host__ __device__
        HIP_vector_type& operator*=(const HIP_vector_type& x) noexcept
        {
            data *= x.data;
            return *this;
        }
        __host__ __device__
        HIP_vector_type& operator/=(const HIP_vector_type& x) noexcept
        {
            data /= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type operator-() noexcept
        {
            auto tmp(*this);
            tmp.data = -tmp.data;
            return tmp;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type operator~() noexcept
        {
            HIP_vector_type r{*this};
            r.data = ~r.data;
            return r;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator%=(const HIP_vector_type& x) noexcept
        {
            data %= x.data;
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator^=(const HIP_vector_type& x) noexcept
        {
            data ^= x.data;
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator|=(const HIP_vector_type& x) noexcept
        {
            data |= x.data;
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator&=(const HIP_vector_type& x) noexcept
        {
            data &= x.data;
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator>>=(const HIP_vector_type& x) noexcept
        {
            data >>= x.data;
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator<<=(const HIP_vector_type& x) noexcept
        {
            data <<= x.data;
            return *this;
        }
    };


    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator*(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= y;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator/(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator==(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        auto tmp = x.data == y.data;
        for (auto i = 0u; i != n; ++i) if (tmp[i] == 0) return false;
        return true;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator!=(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator<(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        auto tmp = x.data < y.data;
        for (auto i = 0u; i != n; ++i) if (tmp[i] == 0) return false;
        return true;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator>(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return y < x;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator<=(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(y < x);
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator>=(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x < y);
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }

    // TODO: the following are rather dubious in terms of general utility.
    template<typename T, unsigned int n>
    inline
    bool operator||(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        auto tmp = x.data || y.data;
        for (auto i = 0u; i != n; ++i) if (tmp[i] == 0) return false;
        return true;
    }

    template<typename T, unsigned int n>
    inline
    bool operator&&(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        auto tmp = x.data && y.data;
        for (auto i = 0u; i != n; ++i) if (tmp[i] == 0) return false;
        return true;
    }

    #define __MAKE_VECTOR_TYPE__(CUDA_name, T, n) \
        using CUDA_name = HIP_vector_type<T, n>;
#else
    typedef unsigned char uchar1 __NATIVE_VECTOR__(1, unsigned char);
    typedef unsigned char uchar2 __NATIVE_VECTOR__(2, unsigned char);
    typedef unsigned char uchar3 __NATIVE_VECTOR__(3, unsigned char);
    typedef unsigned char uchar4 __NATIVE_VECTOR__(4, unsigned char);

    typedef char char1 __NATIVE_VECTOR__(1, char);
    typedef char char2 __NATIVE_VECTOR__(2, char);
    typedef char char3 __NATIVE_VECTOR__(3, char);
    typedef char char4 __NATIVE_VECTOR__(4, char);

    typedef unsigned short ushort1 __NATIVE_VECTOR__(1, unsigned short);
    typedef unsigned short ushort2 __NATIVE_VECTOR__(2, unsigned short);
    typedef unsigned short ushort3 __NATIVE_VECTOR__(3, unsigned short);
    typedef unsigned short ushort4 __NATIVE_VECTOR__(4, unsigned short);

    typedef short short1 __NATIVE_VECTOR__(1, short);
    typedef short short2 __NATIVE_VECTOR__(2, short);
    typedef short short3 __NATIVE_VECTOR__(3, short);
    typedef short short4 __NATIVE_VECTOR__(4, short);

    typedef unsigned int uint1 __NATIVE_VECTOR__(1, unsigned int);
    typedef unsigned int uint2 __NATIVE_VECTOR__(2, unsigned int);
    typedef unsigned int uint3 __NATIVE_VECTOR__(3, unsigned int);
    typedef unsigned int uint4 __NATIVE_VECTOR__(4, unsigned int);

    typedef int int1 __NATIVE_VECTOR__(1, int);
    typedef int int2 __NATIVE_VECTOR__(2, int);
    typedef int int3 __NATIVE_VECTOR__(3, int);
    typedef int int4 __NATIVE_VECTOR__(4, int);

    typedef unsigned long ulong1 __NATIVE_VECTOR__(1, unsigned long);
    typedef unsigned long ulong2 __NATIVE_VECTOR__(2, unsigned long);
    typedef unsigned long ulong3 __NATIVE_VECTOR__(3, unsigned long);
    typedef unsigned long ulong4 __NATIVE_VECTOR__(4, unsigned long);

    typedef long long1 __NATIVE_VECTOR__(1, long);
    typedef long long2 __NATIVE_VECTOR__(2, long);
    typedef long long3 __NATIVE_VECTOR__(3, long);
    typedef long long4 __NATIVE_VECTOR__(4, long);

    typedef unsigned long long ulonglong1 __NATIVE_VECTOR__(1, unsigned long long);
    typedef unsigned long long ulonglong2 __NATIVE_VECTOR__(2, unsigned long long);
    typedef unsigned long long ulonglong3 __NATIVE_VECTOR__(3, unsigned long long);
    typedef unsigned long long ulonglong4 __NATIVE_VECTOR__(4, unsigned long long);

    typedef long long longlong1 __NATIVE_VECTOR__(1, long long);
    typedef long long longlong2 __NATIVE_VECTOR__(2, long long);
    typedef long long longlong3 __NATIVE_VECTOR__(3, long long);
    typedef long long longlong4 __NATIVE_VECTOR__(4, long long);

    typedef float float1 __NATIVE_VECTOR__(1, float);
    typedef float float2 __NATIVE_VECTOR__(2, float);
    typedef float float3 __NATIVE_VECTOR__(3, float);
    typedef float float4 __NATIVE_VECTOR__(4, float);

    typedef double double1 __NATIVE_VECTOR__(1, double);
    typedef double double2 __NATIVE_VECTOR__(2, double);
    typedef double double3 __NATIVE_VECTOR__(3, double);
    typedef double double4 __NATIVE_VECTOR__(4, double);
#endif

__MAKE_VECTOR_TYPE__(uchar1, unsigned char, 1);
__MAKE_VECTOR_TYPE__(uchar2, unsigned char, 2);
__MAKE_VECTOR_TYPE__(uchar3, unsigned char, 3);
__MAKE_VECTOR_TYPE__(uchar4, unsigned char, 4);

__MAKE_VECTOR_TYPE__(char1, char, 1);
__MAKE_VECTOR_TYPE__(char2, char, 2);
__MAKE_VECTOR_TYPE__(char3, char, 3);
__MAKE_VECTOR_TYPE__(char4, char, 4);

__MAKE_VECTOR_TYPE__(ushort1, unsigned short, 1);
__MAKE_VECTOR_TYPE__(ushort2, unsigned short, 2);
__MAKE_VECTOR_TYPE__(ushort3, unsigned short, 3);
__MAKE_VECTOR_TYPE__(ushort4, unsigned short, 4);

__MAKE_VECTOR_TYPE__(short1, short, 1);
__MAKE_VECTOR_TYPE__(short2, short, 2);
__MAKE_VECTOR_TYPE__(short3, short, 3);
__MAKE_VECTOR_TYPE__(short4, short, 4);

__MAKE_VECTOR_TYPE__(uint1, unsigned int, 1);
__MAKE_VECTOR_TYPE__(uint2, unsigned int, 2);
__MAKE_VECTOR_TYPE__(uint3, unsigned int, 3);
__MAKE_VECTOR_TYPE__(uint4, unsigned int, 4);

__MAKE_VECTOR_TYPE__(int1, int, 1);
__MAKE_VECTOR_TYPE__(int2, int, 2);
__MAKE_VECTOR_TYPE__(int3, int, 3);
__MAKE_VECTOR_TYPE__(int4, int, 4);

__MAKE_VECTOR_TYPE__(ulong1, unsigned long, 1);
__MAKE_VECTOR_TYPE__(ulong2, unsigned long, 2);
__MAKE_VECTOR_TYPE__(ulong3, unsigned long, 3);
__MAKE_VECTOR_TYPE__(ulong4, unsigned long, 4);

__MAKE_VECTOR_TYPE__(long1, long, 1);
__MAKE_VECTOR_TYPE__(long2, long, 2);
__MAKE_VECTOR_TYPE__(long3, long, 3);
__MAKE_VECTOR_TYPE__(long4, long, 4);

__MAKE_VECTOR_TYPE__(ulonglong1, unsigned long long, 1);
__MAKE_VECTOR_TYPE__(ulonglong2, unsigned long long, 2);
__MAKE_VECTOR_TYPE__(ulonglong3, unsigned long long, 3);
__MAKE_VECTOR_TYPE__(ulonglong4, unsigned long long, 4);

__MAKE_VECTOR_TYPE__(longlong1, long long, 1);
__MAKE_VECTOR_TYPE__(longlong2, long long, 2);
__MAKE_VECTOR_TYPE__(longlong3, long long, 3);
__MAKE_VECTOR_TYPE__(longlong4, long long, 4);

__MAKE_VECTOR_TYPE__(float1, float, 1);
__MAKE_VECTOR_TYPE__(float2, float, 2);
__MAKE_VECTOR_TYPE__(float3, float, 3);
__MAKE_VECTOR_TYPE__(float4, float, 4);

__MAKE_VECTOR_TYPE__(double1, double, 1);
__MAKE_VECTOR_TYPE__(double2, double, 2);
__MAKE_VECTOR_TYPE__(double3, double, 3);
__MAKE_VECTOR_TYPE__(double4, double, 4);

#define DECLOP_MAKE_ONE_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x) { return type{x}; }

#define DECLOP_MAKE_TWO_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y) { return type{x, y}; }

#define DECLOP_MAKE_THREE_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y, comp z) { return type{x, y, z}; }

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y, comp z, comp w) { \
        return type{x, y, z, w}; \
    }

DECLOP_MAKE_ONE_COMPONENT(unsigned char, uchar1);
DECLOP_MAKE_TWO_COMPONENT(unsigned char, uchar2);
DECLOP_MAKE_THREE_COMPONENT(unsigned char, uchar3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned char, uchar4);

DECLOP_MAKE_ONE_COMPONENT(signed char, char1);
DECLOP_MAKE_TWO_COMPONENT(signed char, char2);
DECLOP_MAKE_THREE_COMPONENT(signed char, char3);
DECLOP_MAKE_FOUR_COMPONENT(signed char, char4);

DECLOP_MAKE_ONE_COMPONENT(unsigned short, ushort1);
DECLOP_MAKE_TWO_COMPONENT(unsigned short, ushort2);
DECLOP_MAKE_THREE_COMPONENT(unsigned short, ushort3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned short, ushort4);

DECLOP_MAKE_ONE_COMPONENT(signed short, short1);
DECLOP_MAKE_TWO_COMPONENT(signed short, short2);
DECLOP_MAKE_THREE_COMPONENT(signed short, short3);
DECLOP_MAKE_FOUR_COMPONENT(signed short, short4);

DECLOP_MAKE_ONE_COMPONENT(unsigned int, uint1);
DECLOP_MAKE_TWO_COMPONENT(unsigned int, uint2);
DECLOP_MAKE_THREE_COMPONENT(unsigned int, uint3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned int, uint4);

DECLOP_MAKE_ONE_COMPONENT(signed int, int1);
DECLOP_MAKE_TWO_COMPONENT(signed int, int2);
DECLOP_MAKE_THREE_COMPONENT(signed int, int3);
DECLOP_MAKE_FOUR_COMPONENT(signed int, int4);

DECLOP_MAKE_ONE_COMPONENT(float, float1);
DECLOP_MAKE_TWO_COMPONENT(float, float2);
DECLOP_MAKE_THREE_COMPONENT(float, float3);
DECLOP_MAKE_FOUR_COMPONENT(float, float4);

DECLOP_MAKE_ONE_COMPONENT(double, double1);
DECLOP_MAKE_TWO_COMPONENT(double, double2);
DECLOP_MAKE_THREE_COMPONENT(double, double3);
DECLOP_MAKE_FOUR_COMPONENT(double, double4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long, ulong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long, ulong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long, ulong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long, ulong4);

DECLOP_MAKE_ONE_COMPONENT(signed long, long1);
DECLOP_MAKE_TWO_COMPONENT(signed long, long2);
DECLOP_MAKE_THREE_COMPONENT(signed long, long3);
DECLOP_MAKE_FOUR_COMPONENT(signed long, long4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long, ulonglong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long, ulonglong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long, ulonglong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long, ulonglong4);

DECLOP_MAKE_ONE_COMPONENT(signed long, longlong1);
DECLOP_MAKE_TWO_COMPONENT(signed long, longlong2);
DECLOP_MAKE_THREE_COMPONENT(signed long, longlong3);
DECLOP_MAKE_FOUR_COMPONENT(signed long, longlong4);

#endif