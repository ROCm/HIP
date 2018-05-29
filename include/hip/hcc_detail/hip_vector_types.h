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

#if defined(__clang__)
    #define __NATIVE_VECTOR__(n, ...) __attribute__((ext_vector_type(n)))
#elif defined(__GNUC__) // N.B.: GCC does not support .xyzw syntax.
    #define __ROUND_UP_TO_NEXT_POT__(x) \
        (1 << (31 - __builtin_clz(x) + (x > (1 << (31 - __builtin_clz(x))))))
    #define __NATIVE_VECTOR__(n, T) \
        __attribute__((vector_size(__ROUND_UP_TO_NEXT_POT__(n) * sizeof(T))))
#endif

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