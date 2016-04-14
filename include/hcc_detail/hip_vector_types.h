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

/**
 *  @file  hcc_detail/hip_vector_types.h
 *  @brief Defines the different newt vector types for HIP runtime.
 */

#ifndef HIP_VECTOR_TYPES_H
#define HIP_VECTOR_TYPES_H

#if defined (__HCC__) &&  (__hcc_workweek__ < 16032)
#error("This version of HIP requires a newer version of HCC.");
#endif

#if __cplusplus
#include <hc_short_vector.hpp>

using namespace hc::short_vector;
#endif

//-- Signed
// Define char vector types
typedef hc::short_vector::char1 char1;
typedef hc::short_vector::char2 char2;
typedef hc::short_vector::char3 char3;
typedef hc::short_vector::char4 char4;

// Define short vector types
typedef hc::short_vector::short1 short1;
typedef hc::short_vector::short2 short2;
typedef hc::short_vector::short3 short3;
typedef hc::short_vector::short4 short4;

// Define int vector types
typedef hc::short_vector::int1 int1;
typedef hc::short_vector::int2 int2;
typedef hc::short_vector::int3 int3;
typedef hc::short_vector::int4 int4;

// Define long vector types
typedef hc::short_vector::long1 long1;
typedef hc::short_vector::long2 long2;
typedef hc::short_vector::long3 long3;
typedef hc::short_vector::long4 long4;

// Define longlong vector types
typedef hc::short_vector::longlong1 longlong1;
typedef hc::short_vector::longlong2 longlong2;
typedef hc::short_vector::longlong3 longlong3;
typedef hc::short_vector::longlong4 longlong4;


//-- Unsigned
// Define uchar vector types
typedef hc::short_vector::uchar1 uchar1;
typedef hc::short_vector::uchar2 uchar2;
typedef hc::short_vector::uchar3 uchar3;
typedef hc::short_vector::uchar4 uchar4;

// Define ushort vector types
typedef hc::short_vector::ushort1 ushort1;
typedef hc::short_vector::ushort2 ushort2;
typedef hc::short_vector::ushort3 ushort3;
typedef hc::short_vector::ushort4 ushort4;

// Define uint vector types
typedef hc::short_vector::uint1 uint1;
typedef hc::short_vector::uint2 uint2;
typedef hc::short_vector::uint3 uint3;
typedef hc::short_vector::uint4 uint4;

// Define ulong vector types
typedef hc::short_vector::ulong1 ulong1;
typedef hc::short_vector::ulong2 ulong2;
typedef hc::short_vector::ulong3 ulong3;
typedef hc::short_vector::ulong4 ulong4;

// Define ulonglong vector types
typedef hc::short_vector::ulonglong1 ulonglong1;
typedef hc::short_vector::ulonglong2 ulonglong2;
typedef hc::short_vector::ulonglong3 ulonglong3;
typedef hc::short_vector::ulonglong4 ulonglong4;


//-- Floating point
// Define float vector types
typedef hc::short_vector::float1 float1;
typedef hc::short_vector::float2 float2;
typedef hc::short_vector::float3 float3;
typedef hc::short_vector::float4 float4;

// Define double vector types
typedef hc::short_vector::double1 double1;
typedef hc::short_vector::double2 double2;
typedef hc::short_vector::double3 double3;
typedef hc::short_vector::double4 double4;


///---
// Inline functions for creating vector types from basic types
#define ONE_COMPONENT_ACCESS(T, VT) inline VT make_ ##VT (T x) { VT t; t.x = x; return t; };
#define TWO_COMPONENT_ACCESS(T, VT) inline VT make_ ##VT (T x, T y) { VT t; t.x=x; t.y=y; return t; };
#define THREE_COMPONENT_ACCESS(T, VT) inline VT make_ ##VT (T x, T y, T z) { VT t; t.x=x; t.y=y; t.z=z; return t; };
#define FOUR_COMPONENT_ACCESS(T, VT) inline VT make_ ##VT (T x, T y, T z, T w) { VT t; t.x=x; t.y=y; t.z=z; t.w=w; return t; };


//signed:
ONE_COMPONENT_ACCESS  (signed char, char1);
TWO_COMPONENT_ACCESS  (signed char, char2);
THREE_COMPONENT_ACCESS(signed char, char3);
FOUR_COMPONENT_ACCESS (signed char, char4);

ONE_COMPONENT_ACCESS  (short, short1);
TWO_COMPONENT_ACCESS  (short, short2);
THREE_COMPONENT_ACCESS(short, short3);
FOUR_COMPONENT_ACCESS (short, short4);

ONE_COMPONENT_ACCESS  (int, int1);
TWO_COMPONENT_ACCESS  (int, int2);
THREE_COMPONENT_ACCESS(int, int3);
FOUR_COMPONENT_ACCESS (int, int4);

ONE_COMPONENT_ACCESS  (long int, long1);
TWO_COMPONENT_ACCESS  (long int, long2);
THREE_COMPONENT_ACCESS(long int, long3);
FOUR_COMPONENT_ACCESS (long int, long4);

ONE_COMPONENT_ACCESS  (long long int, ulong1);
TWO_COMPONENT_ACCESS  (long long int, ulong2);
THREE_COMPONENT_ACCESS(long long int, ulong3);
FOUR_COMPONENT_ACCESS (long long int, ulong4);

ONE_COMPONENT_ACCESS  (long long int, longlong1);
TWO_COMPONENT_ACCESS  (long long int, longlong2);
THREE_COMPONENT_ACCESS(long long int, longlong3);
FOUR_COMPONENT_ACCESS (long long int, longlong4);


// unsigned:
ONE_COMPONENT_ACCESS  (unsigned char, uchar1);
TWO_COMPONENT_ACCESS  (unsigned char, uchar2);
THREE_COMPONENT_ACCESS(unsigned char, uchar3);
FOUR_COMPONENT_ACCESS (unsigned char, uchar4);

ONE_COMPONENT_ACCESS  (unsigned short, ushort1);
TWO_COMPONENT_ACCESS  (unsigned short, ushort2);
THREE_COMPONENT_ACCESS(unsigned short, ushort3);
FOUR_COMPONENT_ACCESS (unsigned short, ushort4);

ONE_COMPONENT_ACCESS  (unsigned int, uint1);
TWO_COMPONENT_ACCESS  (unsigned int, uint2);
THREE_COMPONENT_ACCESS(unsigned int, uint3);
FOUR_COMPONENT_ACCESS (unsigned int, uint4);

ONE_COMPONENT_ACCESS  (unsigned long int, ulong1);
TWO_COMPONENT_ACCESS  (unsigned long int, ulong2);
THREE_COMPONENT_ACCESS(unsigned long int, ulong3);
FOUR_COMPONENT_ACCESS (unsigned long int, ulong4);

ONE_COMPONENT_ACCESS  (unsigned long long int, ulong1);
TWO_COMPONENT_ACCESS  (unsigned long long int, ulong2);
THREE_COMPONENT_ACCESS(unsigned long long int, ulong3);
FOUR_COMPONENT_ACCESS (unsigned long long int, ulong4);

ONE_COMPONENT_ACCESS  (unsigned long long int, ulonglong1);
TWO_COMPONENT_ACCESS  (unsigned long long int, ulonglong2);
THREE_COMPONENT_ACCESS(unsigned long long int, ulonglong3);
FOUR_COMPONENT_ACCESS (unsigned long long int, ulonglong4);


//Floating point
ONE_COMPONENT_ACCESS  (float, float1);
TWO_COMPONENT_ACCESS  (float, float2);
THREE_COMPONENT_ACCESS(float, float3);
FOUR_COMPONENT_ACCESS (float, float4);

ONE_COMPONENT_ACCESS  (double, double1);
TWO_COMPONENT_ACCESS  (double, double2);
THREE_COMPONENT_ACCESS(double, double3);
FOUR_COMPONENT_ACCESS (double, double4);


#endif

