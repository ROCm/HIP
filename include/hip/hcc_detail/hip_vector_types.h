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

#if defined (__HCC__) &&  (__hcc_workweek__ < 16032)
#error("This version of HIP requires a newer version of HCC.");
#endif

#include "hip/hcc_detail/host_defines.h"

#define MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(type) \
__device__ __host__ type() {} \
__device__ __host__ type(const type& val) : x(val.x) { } \
__device__ __host__ ~type() {} 

#define MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(type) \
__device__ __host__ type() {} \
__device__ __host__ type(const type& val) : x(val.x), y(val.y) { } \
__device__ __host__ ~type() {}

#define MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(type) \
__device__ __host__ type() {} \
__device__ __host__ type(const type& val) : x(val.x), y(val.y), z(val.z) { } \
__device__ __host__ ~type() {} 

#define MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(type) \
__device__ __host__ type() {} \
__device__ __host__ type(const type& val) : x(val.x), y(val.y), z(val.z), w(val.w) { } \
__device__ __host__ ~type() {}

#define MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(type, type1) \
__device__ __host__ type(type1 val) : x(val) {} \

#define MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(type, type1) \
__device__ __host__ type(type1 val) : x(val), y(val) {} \
__device__ __host__ type(type1 val1, type1 val2) : x(val1), y(val2) {} \

#define MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(type, type1) \
__device__ __host__ type(type1 val) : x(val), y(val), z(val) {} \
__device__ __host__ type(type1 val1, type1 val2, type1 val3) : x(val1), y(val2), z(val3) {} \

#define MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(type, type1) \
__device__ __host__ type(type1 val) : x(val), y(val), z(val), w(val) {} \
__device__ __host__ type(type1 val1, type1 val2, type1 val3, type1 val4) : x(val1), y(val2), z(val3), w(val4) {} \

struct uchar1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(uchar1)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uchar1, signed long long)

  #endif
  unsigned char x;

} __attribute__((aligned(1)));

struct uchar2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(uchar2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uchar2, signed long long)
  #endif
  union {
    struct {
      unsigned char x, y;
    };
    unsigned short a;
  };
} __attribute__((aligned(2)));

struct uchar3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(uchar3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uchar3, signed long long)
  #endif
  unsigned char x, y, z;
};

struct uchar4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(uchar4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uchar4, signed long long)
  #endif
  union {
    struct {
      unsigned char x, y, z, w;
    };
    unsigned int a;
  };
} __attribute__((aligned(4)));


struct char1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(char1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(char1, signed long long)
  #endif
  signed char x;
} __attribute__((aligned(1)));

struct char2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(char2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(char2, signed long long)
  #endif
  union {
    struct {
      signed char x, y;
    };
    unsigned short a;
  };
} __attribute__((aligned(2)));

struct char3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(char3)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(char3, signed long long)
  #endif
  signed char x, y, z;
};

struct char4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(char4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(char4, signed long long)
  #endif
  union {
    struct {
      signed char x, y, z, w;
    };
    unsigned int a;
  };
} __attribute__((aligned(4)));



struct ushort1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(ushort1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ushort1, signed long long)
  #endif
  unsigned short x;
} __attribute__((aligned(2)));

struct ushort2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(ushort2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ushort2, signed long long)
  #endif
  union {
    struct {
      unsigned short x, y;
    };
    unsigned int a;
  };
} __attribute__((aligned(4)));

struct ushort3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(ushort3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ushort3, signed long long)
  #endif
  unsigned short x, y, z;
};

struct ushort4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(ushort4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ushort4, signed long long)
  #endif
  union {
    struct {
      unsigned short x, y, z, w;
    };
    unsigned int a, b;
  };
} __attribute__((aligned(8)));

struct short1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(short1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(short1, signed long long)
  #endif
  signed short x;
} __attribute__((aligned(2)));

struct short2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(short2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(short2, signed long long)
  #endif
  union {
    struct {
      signed short x, y;
    };
    unsigned int a;
  };

} __attribute__((aligned(4)));

struct short3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(short3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(short3, signed long long)
  #endif
  signed short x, y, z;
};

struct short4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(short4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(short4, signed long long)
  #endif
  union {
    struct {
      signed short x, y, z, w;
    };
    unsigned int a, b;
  };
} __attribute__((aligned(8)));


struct uint1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(uint1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(uint1, signed long long)
  #endif
  unsigned int x;
} __attribute__((aligned(4)));

struct uint2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(uint2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(uint2, signed long long)
  #endif
  unsigned int x, y;
} __attribute__((aligned(8)));

struct uint3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(uint3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(uint3, signed long long)
  #endif
  unsigned int x, y, z;
};

struct uint4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(uint4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(uint4, signed long long)
  #endif
  unsigned int x, y, z, w;
} __attribute__((aligned(16)));

struct int1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(int1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(int1, signed long long)
  #endif
  signed int x;
} __attribute__((aligned(4)));

struct int2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(int2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(int2, signed long long)
  #endif
  signed int x, y;
} __attribute__((aligned(8)));

struct int3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(int3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(int3, signed long long)
  #endif
  signed int x, y, z;
};

struct int4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(int4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(int4, signed long long)
  #endif
  signed int x, y, z, w;
} __attribute__((aligned(16)));


struct float1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(float1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(float1, signed long long)
  #endif
  float x;
} __attribute__((aligned(4)));

struct float2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(float2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(float2, signed long long)
  #endif
  float x, y;
} __attribute__((aligned(8)));

struct float3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(float3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(float3, signed long long)
  #endif
  float x, y, z;
};

struct float4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(float4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(float4, signed long long)
  #endif
  float x, y, z, w;
} __attribute__((aligned(16)));



struct double1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(double1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(double1, signed long long)
  #endif
  double x;
} __attribute__((aligned(8)));

struct double2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(double2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(double2, signed long long)
  #endif
  double x, y;
} __attribute__((aligned(16)));

struct double3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(double3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(double3, signed long long)
  #endif
  double x, y, z;
};

struct double4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(double4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(double4, signed long long)
  #endif
  double x, y, z, w;
} __attribute__((aligned(32)));


struct ulong1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(ulong1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulong1, signed long long)
  #endif
  unsigned long x;
} __attribute__((aligned(8)));

struct ulong2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(ulong2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulong2, signed long long)
  #endif
  unsigned long x, y;
} __attribute__((aligned(16)));

struct ulong3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(ulong3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulong3, signed long long)
  #endif
  unsigned long x, y, z;
};

struct ulong4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(ulong4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulong4, signed long long)
  #endif
  unsigned long x, y, z, w;
} __attribute__((aligned(32)));


struct long1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(long1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(long1, signed long long)
  #endif
  signed long x;
} __attribute__((aligned(8)));

struct long2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(long2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(long2, signed long long)
  #endif
  signed long x, y;
} __attribute__((aligned(16)));

struct long3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(long3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(long3, signed long long)
  #endif
  signed long x, y, z;
};

struct long4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(long4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(long4, signed long long)
  #endif
  signed long x, y, z, w;
} __attribute__((aligned(32)));


struct ulonglong1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(ulonglong1, signed long long)
  #endif
  unsigned long long x;
} __attribute__((aligned(8)));

struct ulonglong2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(ulonglong2, signed long long)
  #endif
  unsigned long long x, y;
} __attribute__((aligned(16)));

struct ulonglong3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(ulonglong3, signed long long)
  #endif
  unsigned long long x, y, z;
};

struct ulonglong4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(ulonglong4, signed long long)
  #endif
  unsigned long long x, y, z, w;
} __attribute__((aligned(32)));


struct longlong1 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_ONE_COMPONENT(longlong1)

    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, float)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, double)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_ONE_COMPONENT(longlong1, signed long long)
  #endif
  signed long long x;
} __attribute__((aligned(8)));

struct longlong2 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_TWO_COMPONENT(longlong2)

    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, float)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, double)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_TWO_COMPONENT(longlong2, signed long long)
  #endif
  signed long long x, y;
} __attribute__((aligned(16)));

struct longlong3 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_THREE_COMPONENT(longlong3)

    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, float)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, double)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_THREE_COMPONENT(longlong3, signed long long)
  #endif
  signed long long x, y, z;
};

struct longlong4 {
  #ifdef __cplusplus
    public:
    MAKE_DEFAULT_CONSTRUCTOR_FOUR_COMPONENT(longlong4)

    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, unsigned char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, signed char)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, unsigned short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, signed short)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, unsigned int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, signed int)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, float)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, double)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, unsigned long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, signed long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, unsigned long long)
    MAKE_COMPONENT_CONSTRUCTOR_FOUR_COMPONENT(longlong4, signed long long)
  #endif
  signed long x, y, z, w;
} __attribute__((aligned(32)));

#define DECLOP_MAKE_ONE_COMPONENT(comp, type) \
__device__ __host__ static inline struct type make_##type(comp x) { \
  struct type ret; \
  ret.x = x; \
  return ret; \
}

#define DECLOP_MAKE_TWO_COMPONENT(comp, type) \
__device__ __host__ static inline struct type make_##type(comp x, comp y) { \
  struct type ret; \
  ret.x = x; \
  ret.y = y; \
  return ret; \
}

#define DECLOP_MAKE_THREE_COMPONENT(comp, type) \
__device__ __host__ static inline struct type make_##type(comp x, comp y, comp z) { \
  struct type ret; \
  ret.x = x; \
  ret.y = y; \
  ret.z = z; \
  return ret; \
}

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type) \
__device__ __host__ static inline struct type make_##type(comp x, comp y, comp z, comp w) { \
  struct type ret; \
  ret.x = x; \
  ret.y = y; \
  ret.z = z; \
  ret.w = w; \
  return ret; \
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


#if __cplusplus

#define DECLOP_1VAR_2IN_1OUT(type, op) \
__device__ __host__ static inline type operator op (const type& lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs.x op rhs.x; \
  return ret; \
}

#define DECLOP_1VAR_SCALE_PRODUCT(type, type1) \
__device__ __host__ static inline type operator * (const type& lhs, type1 rhs) { \
  type ret; \
  ret.x = lhs.x * rhs; \
  return ret; \
} \
\
__device__ __host__ static inline type operator * (type1 lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs * rhs.x; \
  return ret; \
}

#define DECLOP_1VAR_ASSIGN(type, op) \
__device__ __host__ static inline type& operator op ( type& lhs, const type& rhs) { \
  lhs.x op rhs.x; \
  return lhs; \
}

#define DECLOP_1VAR_PREOP(type, op) \
__device__ __host__ static inline type& operator op (type& val) { \
  op val.x; \
  return val; \
}

#define DECLOP_1VAR_POSTOP(type, op) \
__device__ __host__ static inline type operator op (type& val, int) { \
  type ret; \
  ret.x = val.x; \
  val.x op; \
  return ret; \
}

#define DECLOP_1VAR_COMP(type, op) \
__device__ __host__ static inline bool operator op (type& lhs, type& rhs) { \
  return lhs.x op rhs.x; \
} \
__device__ __host__ static inline bool operator op (const type& lhs, type& rhs) { \
  return lhs.x op rhs.x; \
} \
__device__ __host__ static inline bool operator op (type& lhs, const type& rhs) { \
  return lhs.x op rhs.x ; \
} \
__device__ __host__ static inline bool operator op (const type& lhs, const type& rhs) { \
  return lhs.x op rhs.x ; \
}

#define DECLOP_1VAR_1IN_1OUT(type, op) \
__device__ __host__ static inline type operator op(type& rhs) { \
  type ret; \
  ret.x = op rhs.x; \
  return ret; \
}

#define DECLOP_1VAR_1IN_BOOLOUT(type, op) \
__device__ __host__ static inline bool operator op (type& rhs) { \
  return op rhs.x; \
}

/*
 Two Element Access
*/

#define DECLOP_2VAR_2IN_1OUT(type, op) \
__device__ __host__ static inline type operator op (const type& lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs.x op rhs.x; \
  ret.y = lhs.y op rhs.y; \
  return ret; \
}

#define DECLOP_2VAR_SCALE_PRODUCT(type, type1) \
__device__ __host__ static inline type operator * (const type& lhs, type1 rhs) { \
  type ret; \
  ret.x = lhs.x * rhs; \
  ret.y = lhs.y * rhs; \
  return ret; \
} \
\
__device__ __host__ static inline type operator * (type1 lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs * rhs.x; \
  ret.y = lhs * rhs.y; \
  return ret; \
}

#define DECLOP_2VAR_ASSIGN(type, op) \
__device__ __host__ static inline type& operator op ( type& lhs, const type& rhs) { \
  lhs.x op rhs.x; \
  lhs.y op rhs.y; \
  return lhs; \
}

#define DECLOP_2VAR_PREOP(type, op) \
__device__ __host__ static inline type& operator op (type& val) { \
  op val.x; \
  op val.y; \
  return val; \
}

#define DECLOP_2VAR_POSTOP(type, op) \
__device__ __host__ static inline type operator op (type& val, int) { \
  type ret; \
  ret.x = val.x; \
  ret.y = val.y; \
  val.x op; \
  val.y op; \
  return ret; \
}

#define DECLOP_2VAR_COMP(type, op) \
__device__ __host__ static inline bool operator op (type& lhs, type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y); \
} \
__device__ __host__ static inline bool operator op (const type& lhs, type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y); \
} \
__device__ __host__ static inline bool operator op (type& lhs, const type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y); \
} \
__device__ __host__ static inline bool operator op (const type& lhs, const type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y); \
}

#define DECLOP_2VAR_1IN_1OUT(type, op) \
__device__ __host__ static inline type operator op(type &rhs) { \
  type ret; \
  ret.x = op rhs.x; \
  ret.y = op rhs.y; \
  return ret; \
}

#define DECLOP_2VAR_1IN_BOOLOUT(type, op) \
__device__ __host__ static inline bool operator op (type &rhs) { \
  return (op rhs.x) && (op rhs.y); \
}


/*
 Three Element Access
*/

#define DECLOP_3VAR_2IN_1OUT(type, op) \
__device__ __host__ static inline type operator op (const type& lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs.x op rhs.x; \
  ret.y = lhs.y op rhs.y; \
  ret.z = lhs.z op rhs.z; \
  return ret; \
}

#define DECLOP_3VAR_SCALE_PRODUCT(type, type1) \
__device__ __host__ static inline type operator * (const type& lhs, type1 rhs) { \
  type ret; \
  ret.x = lhs.x * rhs; \
  ret.y = lhs.y * rhs; \
  ret.z = lhs.z * rhs; \
  return ret; \
} \
\
__device__ __host__ static inline type operator * (type1 lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs * rhs.x; \
  ret.y = lhs * rhs.y; \
  ret.z = lhs * rhs.z; \
  return ret; \
}

#define DECLOP_3VAR_ASSIGN(type, op) \
__device__ __host__ static inline type& operator op ( type& lhs, const type& rhs) { \
  lhs.x op rhs.x; \
  lhs.y op rhs.y; \
  lhs.z op rhs.z; \
  return lhs; \
}

#define DECLOP_3VAR_PREOP(type, op) \
__device__ __host__ static inline type& operator op (type& val) { \
  op val.x; \
  op val.y; \
  op val.z; \
  return val; \
}

#define DECLOP_3VAR_POSTOP(type, op) \
__device__ __host__ static inline type operator op (type& val, int) { \
  type ret; \
  ret.x = val.x; \
  ret.y = val.y; \
  ret.z = val.z; \
  val.x op; \
  val.y op; \
  val.z op; \
  return ret; \
}

#define DECLOP_3VAR_COMP(type, op) \
__device__ __host__ static inline bool operator op (type& lhs, type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z); \
} \
__device__ __host__ static inline bool operator op (const type& lhs, type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z); \
} \
__device__ __host__ static inline bool operator op (type& lhs, const type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z); \
} \
__device__ __host__ static inline bool operator op (const type& lhs, const type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z); \
} \

#define DECLOP_3VAR_1IN_1OUT(type, op) \
__device__ __host__ static inline type operator op(type &rhs) { \
  type ret; \
  ret.x = op rhs.x; \
  ret.y = op rhs.y; \
  ret.z = op rhs.z; \
  return ret; \
}

#define DECLOP_3VAR_1IN_BOOLOUT(type, op) \
__device__ __host__ static inline bool operator op (type &rhs) { \
  return (op rhs.x) && (op rhs.y) && (op rhs.z); \
}


/*
 Four Element Access
*/

#define DECLOP_4VAR_2IN_1OUT(type, op) \
__device__ __host__ static inline type operator op ( const type& lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs.x op rhs.x; \
  ret.y = lhs.y op rhs.y; \
  ret.z = lhs.z op rhs.z; \
  ret.w = lhs.w op rhs.w; \
  return ret; \
}

#define DECLOP_4VAR_SCALE_PRODUCT(type, type1) \
__device__ __host__ static inline type operator * (const type& lhs, type1 rhs) { \
  type ret; \
  ret.x = lhs.x * rhs; \
  ret.y = lhs.y * rhs; \
  ret.z = lhs.z * rhs; \
  ret.w = lhs.w * rhs; \
  return ret; \
} \
\
__device__ __host__ static inline type operator * (type1 lhs, const type& rhs) { \
  type ret; \
  ret.x = lhs * rhs.x; \
  ret.y = lhs * rhs.y; \
  ret.z = lhs * rhs.z; \
  ret.w = lhs * rhs.w; \
  return ret; \
}

#define DECLOP_4VAR_ASSIGN(type, op) \
__device__ __host__ static inline type& operator op ( type& lhs, const type& rhs) { \
  lhs.x op rhs.x; \
  lhs.y op rhs.y; \
  lhs.z op rhs.z; \
  lhs.w op rhs.w; \
  return lhs; \
}

#define DECLOP_4VAR_PREOP(type, op) \
__device__ __host__ static inline type& operator op (type& val) { \
  op val.x; \
  op val.y; \
  op val.z; \
  op val.w; \
  return val; \
}

#define DECLOP_4VAR_POSTOP(type, op) \
__device__ __host__ static inline type operator op (type& val, int) { \
  type ret; \
  ret.x = val.x; \
  ret.y = val.y; \
  ret.z = val.z; \
  ret.w = val.w; \
  val.x op; \
  val.y op; \
  val.z op; \
  val.w op; \
  return ret; \
}

#define DECLOP_4VAR_COMP(type, op) \
__device__ __host__ static inline bool operator op (type& lhs, type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z) && (lhs.w op rhs.w); \
} \
__device__ __host__ static inline bool operator op (const type& lhs, type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z) && (lhs.w op rhs.w); \
} \
__device__ __host__ static inline bool operator op (type& lhs, const type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z) && (lhs.w op rhs.w); \
} \
__device__ __host__ static inline bool operator op (const type& lhs, const type& rhs) { \
  return (lhs.x op rhs.x) && (lhs.y op rhs.y) && (lhs.z op rhs.z) && (lhs.w op rhs.w); \
}

#define DECLOP_4VAR_1IN_1OUT(type, op) \
__device__ __host__ static inline type operator op(type &rhs) { \
  type ret; \
  ret.x = op rhs.x; \
  ret.y = op rhs.y; \
  ret.z = op rhs.z; \
  ret.w = op rhs.w; \
  return ret; \
}

#define DECLOP_4VAR_1IN_BOOLOUT(type, op) \
__device__ __host__ static inline bool operator op (type &rhs) { \
  return (op rhs.x) && (op rhs.y) && (op rhs.z) && (op rhs.w); \
}


/*
Overloading operators
*/

// UNSIGNED CHAR1

DECLOP_1VAR_2IN_1OUT(uchar1, +)
DECLOP_1VAR_2IN_1OUT(uchar1, -)
DECLOP_1VAR_2IN_1OUT(uchar1, *)
DECLOP_1VAR_2IN_1OUT(uchar1, /)
DECLOP_1VAR_2IN_1OUT(uchar1, %)
DECLOP_1VAR_2IN_1OUT(uchar1, &)
DECLOP_1VAR_2IN_1OUT(uchar1, |)
DECLOP_1VAR_2IN_1OUT(uchar1, ^)
DECLOP_1VAR_2IN_1OUT(uchar1, <<)
DECLOP_1VAR_2IN_1OUT(uchar1, >>)


DECLOP_1VAR_ASSIGN(uchar1, +=)
DECLOP_1VAR_ASSIGN(uchar1, -=)
DECLOP_1VAR_ASSIGN(uchar1, *=)
DECLOP_1VAR_ASSIGN(uchar1, /=)
DECLOP_1VAR_ASSIGN(uchar1, %=)
DECLOP_1VAR_ASSIGN(uchar1, &=)
DECLOP_1VAR_ASSIGN(uchar1, |=)
DECLOP_1VAR_ASSIGN(uchar1, ^=)
DECLOP_1VAR_ASSIGN(uchar1, <<=)
DECLOP_1VAR_ASSIGN(uchar1, >>=)

DECLOP_1VAR_PREOP(uchar1, ++)
DECLOP_1VAR_PREOP(uchar1, --)

DECLOP_1VAR_POSTOP(uchar1, ++)
DECLOP_1VAR_POSTOP(uchar1, --)

DECLOP_1VAR_COMP(uchar1, ==)
DECLOP_1VAR_COMP(uchar1, !=)
DECLOP_1VAR_COMP(uchar1, <)
DECLOP_1VAR_COMP(uchar1, >)
DECLOP_1VAR_COMP(uchar1, <=)
DECLOP_1VAR_COMP(uchar1, >=)

DECLOP_1VAR_COMP(uchar1, &&)
DECLOP_1VAR_COMP(uchar1, ||)

DECLOP_1VAR_1IN_1OUT(uchar1, ~)
DECLOP_1VAR_1IN_BOOLOUT(uchar1, !)

DECLOP_1VAR_SCALE_PRODUCT(uchar1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, float)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, double)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(uchar1, signed long long)

// UNSIGNED CHAR2

DECLOP_2VAR_2IN_1OUT(uchar2, +)
DECLOP_2VAR_2IN_1OUT(uchar2, -)
DECLOP_2VAR_2IN_1OUT(uchar2, *)
DECLOP_2VAR_2IN_1OUT(uchar2, /)
DECLOP_2VAR_2IN_1OUT(uchar2, %)
DECLOP_2VAR_2IN_1OUT(uchar2, &)
DECLOP_2VAR_2IN_1OUT(uchar2, |)
DECLOP_2VAR_2IN_1OUT(uchar2, ^)
DECLOP_2VAR_2IN_1OUT(uchar2, <<)
DECLOP_2VAR_2IN_1OUT(uchar2, >>)

DECLOP_2VAR_ASSIGN(uchar2, +=)
DECLOP_2VAR_ASSIGN(uchar2, -=)
DECLOP_2VAR_ASSIGN(uchar2, *=)
DECLOP_2VAR_ASSIGN(uchar2, /=)
DECLOP_2VAR_ASSIGN(uchar2, %=)
DECLOP_2VAR_ASSIGN(uchar2, &=)
DECLOP_2VAR_ASSIGN(uchar2, |=)
DECLOP_2VAR_ASSIGN(uchar2, ^=)
DECLOP_2VAR_ASSIGN(uchar2, <<=)
DECLOP_2VAR_ASSIGN(uchar2, >>=)

DECLOP_2VAR_PREOP(uchar2, ++)
DECLOP_2VAR_PREOP(uchar2, --)

DECLOP_2VAR_POSTOP(uchar2, ++)
DECLOP_2VAR_POSTOP(uchar2, --)

DECLOP_2VAR_COMP(uchar2, ==)
DECLOP_2VAR_COMP(uchar2, !=)
DECLOP_2VAR_COMP(uchar2, <)
DECLOP_2VAR_COMP(uchar2, >)
DECLOP_2VAR_COMP(uchar2, <=)
DECLOP_2VAR_COMP(uchar2, >=)

DECLOP_2VAR_COMP(uchar2, &&)
DECLOP_2VAR_COMP(uchar2, ||)

DECLOP_2VAR_1IN_1OUT(uchar2, ~)
DECLOP_2VAR_1IN_BOOLOUT(uchar2, !)

DECLOP_2VAR_SCALE_PRODUCT(uchar2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, float)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, double)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(uchar2, signed long long)

// UNSIGNED CHAR3

DECLOP_3VAR_2IN_1OUT(uchar3, +)
DECLOP_3VAR_2IN_1OUT(uchar3, -)
DECLOP_3VAR_2IN_1OUT(uchar3, *)
DECLOP_3VAR_2IN_1OUT(uchar3, /)
DECLOP_3VAR_2IN_1OUT(uchar3, %)
DECLOP_3VAR_2IN_1OUT(uchar3, &)
DECLOP_3VAR_2IN_1OUT(uchar3, |)
DECLOP_3VAR_2IN_1OUT(uchar3, ^)
DECLOP_3VAR_2IN_1OUT(uchar3, <<)
DECLOP_3VAR_2IN_1OUT(uchar3, >>)

DECLOP_3VAR_ASSIGN(uchar3, +=)
DECLOP_3VAR_ASSIGN(uchar3, -=)
DECLOP_3VAR_ASSIGN(uchar3, *=)
DECLOP_3VAR_ASSIGN(uchar3, /=)
DECLOP_3VAR_ASSIGN(uchar3, %=)
DECLOP_3VAR_ASSIGN(uchar3, &=)
DECLOP_3VAR_ASSIGN(uchar3, |=)
DECLOP_3VAR_ASSIGN(uchar3, ^=)
DECLOP_3VAR_ASSIGN(uchar3, <<=)
DECLOP_3VAR_ASSIGN(uchar3, >>=)

DECLOP_3VAR_PREOP(uchar3, ++)
DECLOP_3VAR_PREOP(uchar3, --)

DECLOP_3VAR_POSTOP(uchar3, ++)
DECLOP_3VAR_POSTOP(uchar3, --)

DECLOP_3VAR_COMP(uchar3, ==)
DECLOP_3VAR_COMP(uchar3, !=)
DECLOP_3VAR_COMP(uchar3, <)
DECLOP_3VAR_COMP(uchar3, >)
DECLOP_3VAR_COMP(uchar3, <=)
DECLOP_3VAR_COMP(uchar3, >=)

DECLOP_3VAR_COMP(uchar3, &&)
DECLOP_3VAR_COMP(uchar3, ||)

DECLOP_3VAR_1IN_1OUT(uchar3, ~)
DECLOP_3VAR_1IN_BOOLOUT(uchar3, !)

DECLOP_3VAR_SCALE_PRODUCT(uchar3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, float)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, double)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(uchar3, signed long long)

// UNSIGNED CHAR4

DECLOP_4VAR_2IN_1OUT(uchar4, +)
DECLOP_4VAR_2IN_1OUT(uchar4, -)
DECLOP_4VAR_2IN_1OUT(uchar4, *)
DECLOP_4VAR_2IN_1OUT(uchar4, /)
DECLOP_4VAR_2IN_1OUT(uchar4, %)
DECLOP_4VAR_2IN_1OUT(uchar4, &)
DECLOP_4VAR_2IN_1OUT(uchar4, |)
DECLOP_4VAR_2IN_1OUT(uchar4, ^)
DECLOP_4VAR_2IN_1OUT(uchar4, <<)
DECLOP_4VAR_2IN_1OUT(uchar4, >>)

DECLOP_4VAR_ASSIGN(uchar4, +=)
DECLOP_4VAR_ASSIGN(uchar4, -=)
DECLOP_4VAR_ASSIGN(uchar4, *=)
DECLOP_4VAR_ASSIGN(uchar4, /=)
DECLOP_4VAR_ASSIGN(uchar4, %=)
DECLOP_4VAR_ASSIGN(uchar4, &=)
DECLOP_4VAR_ASSIGN(uchar4, |=)
DECLOP_4VAR_ASSIGN(uchar4, ^=)
DECLOP_4VAR_ASSIGN(uchar4, <<=)
DECLOP_4VAR_ASSIGN(uchar4, >>=)

DECLOP_4VAR_PREOP(uchar4, ++)
DECLOP_4VAR_PREOP(uchar4, --)

DECLOP_4VAR_POSTOP(uchar4, ++)
DECLOP_4VAR_POSTOP(uchar4, --)

DECLOP_4VAR_COMP(uchar4, ==)
DECLOP_4VAR_COMP(uchar4, !=)
DECLOP_4VAR_COMP(uchar4, <)
DECLOP_4VAR_COMP(uchar4, >)
DECLOP_4VAR_COMP(uchar4, <=)
DECLOP_4VAR_COMP(uchar4, >=)

DECLOP_4VAR_COMP(uchar4, &&)
DECLOP_4VAR_COMP(uchar4, ||)

DECLOP_4VAR_1IN_1OUT(uchar4, ~)
DECLOP_4VAR_1IN_BOOLOUT(uchar4, !)

DECLOP_4VAR_SCALE_PRODUCT(uchar4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, float)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, double)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(uchar4, signed long long)

// SIGNED CHAR1

DECLOP_1VAR_2IN_1OUT(char1, +)
DECLOP_1VAR_2IN_1OUT(char1, -)
DECLOP_1VAR_2IN_1OUT(char1, *)
DECLOP_1VAR_2IN_1OUT(char1, /)
DECLOP_1VAR_2IN_1OUT(char1, %)
DECLOP_1VAR_2IN_1OUT(char1, &)
DECLOP_1VAR_2IN_1OUT(char1, |)
DECLOP_1VAR_2IN_1OUT(char1, ^)
DECLOP_1VAR_2IN_1OUT(char1, <<)
DECLOP_1VAR_2IN_1OUT(char1, >>)


DECLOP_1VAR_ASSIGN(char1, +=)
DECLOP_1VAR_ASSIGN(char1, -=)
DECLOP_1VAR_ASSIGN(char1, *=)
DECLOP_1VAR_ASSIGN(char1, /=)
DECLOP_1VAR_ASSIGN(char1, %=)
DECLOP_1VAR_ASSIGN(char1, &=)
DECLOP_1VAR_ASSIGN(char1, |=)
DECLOP_1VAR_ASSIGN(char1, ^=)
DECLOP_1VAR_ASSIGN(char1, <<=)
DECLOP_1VAR_ASSIGN(char1, >>=)

DECLOP_1VAR_PREOP(char1, ++)
DECLOP_1VAR_PREOP(char1, --)

DECLOP_1VAR_POSTOP(char1, ++)
DECLOP_1VAR_POSTOP(char1, --)

DECLOP_1VAR_COMP(char1, ==)
DECLOP_1VAR_COMP(char1, !=)
DECLOP_1VAR_COMP(char1, <)
DECLOP_1VAR_COMP(char1, >)
DECLOP_1VAR_COMP(char1, <=)
DECLOP_1VAR_COMP(char1, >=)

DECLOP_1VAR_COMP(char1, &&)
DECLOP_1VAR_COMP(char1, ||)

DECLOP_1VAR_1IN_1OUT(char1, ~)
DECLOP_1VAR_1IN_BOOLOUT(char1, !)

DECLOP_1VAR_SCALE_PRODUCT(char1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(char1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(char1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(char1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(char1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(char1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(char1, float)
DECLOP_1VAR_SCALE_PRODUCT(char1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(char1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(char1, double)
DECLOP_1VAR_SCALE_PRODUCT(char1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(char1, signed long long)

// SIGNED CHAR2

DECLOP_2VAR_2IN_1OUT(char2, +)
DECLOP_2VAR_2IN_1OUT(char2, -)
DECLOP_2VAR_2IN_1OUT(char2, *)
DECLOP_2VAR_2IN_1OUT(char2, /)
DECLOP_2VAR_2IN_1OUT(char2, %)
DECLOP_2VAR_2IN_1OUT(char2, &)
DECLOP_2VAR_2IN_1OUT(char2, |)
DECLOP_2VAR_2IN_1OUT(char2, ^)
DECLOP_2VAR_2IN_1OUT(char2, <<)
DECLOP_2VAR_2IN_1OUT(char2, >>)

DECLOP_2VAR_ASSIGN(char2, +=)
DECLOP_2VAR_ASSIGN(char2, -=)
DECLOP_2VAR_ASSIGN(char2, *=)
DECLOP_2VAR_ASSIGN(char2, /=)
DECLOP_2VAR_ASSIGN(char2, %=)
DECLOP_2VAR_ASSIGN(char2, &=)
DECLOP_2VAR_ASSIGN(char2, |=)
DECLOP_2VAR_ASSIGN(char2, ^=)
DECLOP_2VAR_ASSIGN(char2, <<=)
DECLOP_2VAR_ASSIGN(char2, >>=)

DECLOP_2VAR_PREOP(char2, ++)
DECLOP_2VAR_PREOP(char2, --)

DECLOP_2VAR_POSTOP(char2, ++)
DECLOP_2VAR_POSTOP(char2, --)

DECLOP_2VAR_COMP(char2, ==)
DECLOP_2VAR_COMP(char2, !=)
DECLOP_2VAR_COMP(char2, <)
DECLOP_2VAR_COMP(char2, >)
DECLOP_2VAR_COMP(char2, <=)
DECLOP_2VAR_COMP(char2, >=)

DECLOP_2VAR_COMP(char2, &&)
DECLOP_2VAR_COMP(char2, ||)

DECLOP_2VAR_1IN_1OUT(char2, ~)
DECLOP_2VAR_1IN_BOOLOUT(char2, !)

DECLOP_2VAR_SCALE_PRODUCT(char2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(char2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(char2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(char2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(char2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(char2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(char2, float)
DECLOP_2VAR_SCALE_PRODUCT(char2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(char2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(char2, double)
DECLOP_2VAR_SCALE_PRODUCT(char2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(char2, signed long long)

// SIGNED CHAR3

DECLOP_3VAR_2IN_1OUT(char3, +)
DECLOP_3VAR_2IN_1OUT(char3, -)
DECLOP_3VAR_2IN_1OUT(char3, *)
DECLOP_3VAR_2IN_1OUT(char3, /)
DECLOP_3VAR_2IN_1OUT(char3, %)
DECLOP_3VAR_2IN_1OUT(char3, &)
DECLOP_3VAR_2IN_1OUT(char3, |)
DECLOP_3VAR_2IN_1OUT(char3, ^)
DECLOP_3VAR_2IN_1OUT(char3, <<)
DECLOP_3VAR_2IN_1OUT(char3, >>)

DECLOP_3VAR_ASSIGN(char3, +=)
DECLOP_3VAR_ASSIGN(char3, -=)
DECLOP_3VAR_ASSIGN(char3, *=)
DECLOP_3VAR_ASSIGN(char3, /=)
DECLOP_3VAR_ASSIGN(char3, %=)
DECLOP_3VAR_ASSIGN(char3, &=)
DECLOP_3VAR_ASSIGN(char3, |=)
DECLOP_3VAR_ASSIGN(char3, ^=)
DECLOP_3VAR_ASSIGN(char3, <<=)
DECLOP_3VAR_ASSIGN(char3, >>=)

DECLOP_3VAR_PREOP(char3, ++)
DECLOP_3VAR_PREOP(char3, --)

DECLOP_3VAR_POSTOP(char3, ++)
DECLOP_3VAR_POSTOP(char3, --)

DECLOP_3VAR_COMP(char3, ==)
DECLOP_3VAR_COMP(char3, !=)
DECLOP_3VAR_COMP(char3, <)
DECLOP_3VAR_COMP(char3, >)
DECLOP_3VAR_COMP(char3, <=)
DECLOP_3VAR_COMP(char3, >=)

DECLOP_3VAR_COMP(char3, &&)
DECLOP_3VAR_COMP(char3, ||)

DECLOP_3VAR_1IN_1OUT(char3, ~)
DECLOP_3VAR_1IN_BOOLOUT(char3, !)

DECLOP_3VAR_SCALE_PRODUCT(char3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(char3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(char3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(char3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(char3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(char3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(char3, float)
DECLOP_3VAR_SCALE_PRODUCT(char3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(char3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(char3, double)
DECLOP_3VAR_SCALE_PRODUCT(char3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(char3, signed long long)

// SIGNED CHAR4

DECLOP_4VAR_2IN_1OUT(char4, +)
DECLOP_4VAR_2IN_1OUT(char4, -)
DECLOP_4VAR_2IN_1OUT(char4, *)
DECLOP_4VAR_2IN_1OUT(char4, /)
DECLOP_4VAR_2IN_1OUT(char4, %)
DECLOP_4VAR_2IN_1OUT(char4, &)
DECLOP_4VAR_2IN_1OUT(char4, |)
DECLOP_4VAR_2IN_1OUT(char4, ^)
DECLOP_4VAR_2IN_1OUT(char4, <<)
DECLOP_4VAR_2IN_1OUT(char4, >>)

DECLOP_4VAR_ASSIGN(char4, +=)
DECLOP_4VAR_ASSIGN(char4, -=)
DECLOP_4VAR_ASSIGN(char4, *=)
DECLOP_4VAR_ASSIGN(char4, /=)
DECLOP_4VAR_ASSIGN(char4, %=)
DECLOP_4VAR_ASSIGN(char4, &=)
DECLOP_4VAR_ASSIGN(char4, |=)
DECLOP_4VAR_ASSIGN(char4, ^=)
DECLOP_4VAR_ASSIGN(char4, <<=)
DECLOP_4VAR_ASSIGN(char4, >>=)

DECLOP_4VAR_PREOP(char4, ++)
DECLOP_4VAR_PREOP(char4, --)

DECLOP_4VAR_POSTOP(char4, ++)
DECLOP_4VAR_POSTOP(char4, --)

DECLOP_4VAR_COMP(char4, ==)
DECLOP_4VAR_COMP(char4, !=)
DECLOP_4VAR_COMP(char4, <)
DECLOP_4VAR_COMP(char4, >)
DECLOP_4VAR_COMP(char4, <=)
DECLOP_4VAR_COMP(char4, >=)

DECLOP_4VAR_COMP(char4, &&)
DECLOP_4VAR_COMP(char4, ||)

DECLOP_4VAR_1IN_1OUT(char4, ~)
DECLOP_4VAR_1IN_BOOLOUT(char4, !)

DECLOP_4VAR_SCALE_PRODUCT(char4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(char4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(char4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(char4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(char4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(char4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(char4, float)
DECLOP_4VAR_SCALE_PRODUCT(char4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(char4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(char4, double)
DECLOP_4VAR_SCALE_PRODUCT(char4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(char4, signed long long)

// UNSIGNED SHORT1

DECLOP_1VAR_2IN_1OUT(ushort1, +)
DECLOP_1VAR_2IN_1OUT(ushort1, -)
DECLOP_1VAR_2IN_1OUT(ushort1, *)
DECLOP_1VAR_2IN_1OUT(ushort1, /)
DECLOP_1VAR_2IN_1OUT(ushort1, %)
DECLOP_1VAR_2IN_1OUT(ushort1, &)
DECLOP_1VAR_2IN_1OUT(ushort1, |)
DECLOP_1VAR_2IN_1OUT(ushort1, ^)
DECLOP_1VAR_2IN_1OUT(ushort1, <<)
DECLOP_1VAR_2IN_1OUT(ushort1, >>)


DECLOP_1VAR_ASSIGN(ushort1, +=)
DECLOP_1VAR_ASSIGN(ushort1, -=)
DECLOP_1VAR_ASSIGN(ushort1, *=)
DECLOP_1VAR_ASSIGN(ushort1, /=)
DECLOP_1VAR_ASSIGN(ushort1, %=)
DECLOP_1VAR_ASSIGN(ushort1, &=)
DECLOP_1VAR_ASSIGN(ushort1, |=)
DECLOP_1VAR_ASSIGN(ushort1, ^=)
DECLOP_1VAR_ASSIGN(ushort1, <<=)
DECLOP_1VAR_ASSIGN(ushort1, >>=)

DECLOP_1VAR_PREOP(ushort1, ++)
DECLOP_1VAR_PREOP(ushort1, --)

DECLOP_1VAR_POSTOP(ushort1, ++)
DECLOP_1VAR_POSTOP(ushort1, --)

DECLOP_1VAR_COMP(ushort1, ==)
DECLOP_1VAR_COMP(ushort1, !=)
DECLOP_1VAR_COMP(ushort1, <)
DECLOP_1VAR_COMP(ushort1, >)
DECLOP_1VAR_COMP(ushort1, <=)
DECLOP_1VAR_COMP(ushort1, >=)

DECLOP_1VAR_COMP(ushort1, &&)
DECLOP_1VAR_COMP(ushort1, ||)

DECLOP_1VAR_1IN_1OUT(ushort1, ~)
DECLOP_1VAR_1IN_BOOLOUT(ushort1, !)

DECLOP_1VAR_SCALE_PRODUCT(ushort1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, float)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, double)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(ushort1, signed long long)

// UNSIGNED SHORT2

DECLOP_2VAR_2IN_1OUT(ushort2, +)
DECLOP_2VAR_2IN_1OUT(ushort2, -)
DECLOP_2VAR_2IN_1OUT(ushort2, *)
DECLOP_2VAR_2IN_1OUT(ushort2, /)
DECLOP_2VAR_2IN_1OUT(ushort2, %)
DECLOP_2VAR_2IN_1OUT(ushort2, &)
DECLOP_2VAR_2IN_1OUT(ushort2, |)
DECLOP_2VAR_2IN_1OUT(ushort2, ^)
DECLOP_2VAR_2IN_1OUT(ushort2, <<)
DECLOP_2VAR_2IN_1OUT(ushort2, >>)

DECLOP_2VAR_ASSIGN(ushort2, +=)
DECLOP_2VAR_ASSIGN(ushort2, -=)
DECLOP_2VAR_ASSIGN(ushort2, *=)
DECLOP_2VAR_ASSIGN(ushort2, /=)
DECLOP_2VAR_ASSIGN(ushort2, %=)
DECLOP_2VAR_ASSIGN(ushort2, &=)
DECLOP_2VAR_ASSIGN(ushort2, |=)
DECLOP_2VAR_ASSIGN(ushort2, ^=)
DECLOP_2VAR_ASSIGN(ushort2, <<=)
DECLOP_2VAR_ASSIGN(ushort2, >>=)

DECLOP_2VAR_PREOP(ushort2, ++)
DECLOP_2VAR_PREOP(ushort2, --)

DECLOP_2VAR_POSTOP(ushort2, ++)
DECLOP_2VAR_POSTOP(ushort2, --)

DECLOP_2VAR_COMP(ushort2, ==)
DECLOP_2VAR_COMP(ushort2, !=)
DECLOP_2VAR_COMP(ushort2, <)
DECLOP_2VAR_COMP(ushort2, >)
DECLOP_2VAR_COMP(ushort2, <=)
DECLOP_2VAR_COMP(ushort2, >=)

DECLOP_2VAR_COMP(ushort2, &&)
DECLOP_2VAR_COMP(ushort2, ||)

DECLOP_2VAR_1IN_1OUT(ushort2, ~)
DECLOP_2VAR_1IN_BOOLOUT(ushort2, !)

DECLOP_2VAR_SCALE_PRODUCT(ushort2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, float)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, double)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(ushort2, signed long long)

// UNSIGNED SHORT3

DECLOP_3VAR_2IN_1OUT(ushort3, +)
DECLOP_3VAR_2IN_1OUT(ushort3, -)
DECLOP_3VAR_2IN_1OUT(ushort3, *)
DECLOP_3VAR_2IN_1OUT(ushort3, /)
DECLOP_3VAR_2IN_1OUT(ushort3, %)
DECLOP_3VAR_2IN_1OUT(ushort3, &)
DECLOP_3VAR_2IN_1OUT(ushort3, |)
DECLOP_3VAR_2IN_1OUT(ushort3, ^)
DECLOP_3VAR_2IN_1OUT(ushort3, <<)
DECLOP_3VAR_2IN_1OUT(ushort3, >>)

DECLOP_3VAR_ASSIGN(ushort3, +=)
DECLOP_3VAR_ASSIGN(ushort3, -=)
DECLOP_3VAR_ASSIGN(ushort3, *=)
DECLOP_3VAR_ASSIGN(ushort3, /=)
DECLOP_3VAR_ASSIGN(ushort3, %=)
DECLOP_3VAR_ASSIGN(ushort3, &=)
DECLOP_3VAR_ASSIGN(ushort3, |=)
DECLOP_3VAR_ASSIGN(ushort3, ^=)
DECLOP_3VAR_ASSIGN(ushort3, <<=)
DECLOP_3VAR_ASSIGN(ushort3, >>=)

DECLOP_3VAR_PREOP(ushort3, ++)
DECLOP_3VAR_PREOP(ushort3, --)

DECLOP_3VAR_POSTOP(ushort3, ++)
DECLOP_3VAR_POSTOP(ushort3, --)

DECLOP_3VAR_COMP(ushort3, ==)
DECLOP_3VAR_COMP(ushort3, !=)
DECLOP_3VAR_COMP(ushort3, <)
DECLOP_3VAR_COMP(ushort3, >)
DECLOP_3VAR_COMP(ushort3, <=)
DECLOP_3VAR_COMP(ushort3, >=)

DECLOP_3VAR_COMP(ushort3, &&)
DECLOP_3VAR_COMP(ushort3, ||)

DECLOP_3VAR_1IN_1OUT(ushort3, ~)
DECLOP_3VAR_1IN_BOOLOUT(ushort3, !)

DECLOP_3VAR_SCALE_PRODUCT(ushort3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, float)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, double)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(ushort3, signed long long)

// UNSIGNED SHORT4

DECLOP_4VAR_2IN_1OUT(ushort4, +)
DECLOP_4VAR_2IN_1OUT(ushort4, -)
DECLOP_4VAR_2IN_1OUT(ushort4, *)
DECLOP_4VAR_2IN_1OUT(ushort4, /)
DECLOP_4VAR_2IN_1OUT(ushort4, %)
DECLOP_4VAR_2IN_1OUT(ushort4, &)
DECLOP_4VAR_2IN_1OUT(ushort4, |)
DECLOP_4VAR_2IN_1OUT(ushort4, ^)
DECLOP_4VAR_2IN_1OUT(ushort4, <<)
DECLOP_4VAR_2IN_1OUT(ushort4, >>)

DECLOP_4VAR_ASSIGN(ushort4, +=)
DECLOP_4VAR_ASSIGN(ushort4, -=)
DECLOP_4VAR_ASSIGN(ushort4, *=)
DECLOP_4VAR_ASSIGN(ushort4, /=)
DECLOP_4VAR_ASSIGN(ushort4, %=)
DECLOP_4VAR_ASSIGN(ushort4, &=)
DECLOP_4VAR_ASSIGN(ushort4, |=)
DECLOP_4VAR_ASSIGN(ushort4, ^=)
DECLOP_4VAR_ASSIGN(ushort4, <<=)
DECLOP_4VAR_ASSIGN(ushort4, >>=)

DECLOP_4VAR_PREOP(ushort4, ++)
DECLOP_4VAR_PREOP(ushort4, --)

DECLOP_4VAR_POSTOP(ushort4, ++)
DECLOP_4VAR_POSTOP(ushort4, --)

DECLOP_4VAR_COMP(ushort4, ==)
DECLOP_4VAR_COMP(ushort4, !=)
DECLOP_4VAR_COMP(ushort4, <)
DECLOP_4VAR_COMP(ushort4, >)
DECLOP_4VAR_COMP(ushort4, <=)
DECLOP_4VAR_COMP(ushort4, >=)

DECLOP_4VAR_COMP(ushort4, &&)
DECLOP_4VAR_COMP(ushort4, ||)

DECLOP_4VAR_1IN_1OUT(ushort4, ~)
DECLOP_4VAR_1IN_BOOLOUT(ushort4, !)

DECLOP_4VAR_SCALE_PRODUCT(ushort4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, float)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, double)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(ushort4, signed long long)

// SIGNED SHORT1

DECLOP_1VAR_2IN_1OUT(short1, +)
DECLOP_1VAR_2IN_1OUT(short1, -)
DECLOP_1VAR_2IN_1OUT(short1, *)
DECLOP_1VAR_2IN_1OUT(short1, /)
DECLOP_1VAR_2IN_1OUT(short1, %)
DECLOP_1VAR_2IN_1OUT(short1, &)
DECLOP_1VAR_2IN_1OUT(short1, |)
DECLOP_1VAR_2IN_1OUT(short1, ^)
DECLOP_1VAR_2IN_1OUT(short1, <<)
DECLOP_1VAR_2IN_1OUT(short1, >>)


DECLOP_1VAR_ASSIGN(short1, +=)
DECLOP_1VAR_ASSIGN(short1, -=)
DECLOP_1VAR_ASSIGN(short1, *=)
DECLOP_1VAR_ASSIGN(short1, /=)
DECLOP_1VAR_ASSIGN(short1, %=)
DECLOP_1VAR_ASSIGN(short1, &=)
DECLOP_1VAR_ASSIGN(short1, |=)
DECLOP_1VAR_ASSIGN(short1, ^=)
DECLOP_1VAR_ASSIGN(short1, <<=)
DECLOP_1VAR_ASSIGN(short1, >>=)

DECLOP_1VAR_PREOP(short1, ++)
DECLOP_1VAR_PREOP(short1, --)

DECLOP_1VAR_POSTOP(short1, ++)
DECLOP_1VAR_POSTOP(short1, --)

DECLOP_1VAR_COMP(short1, ==)
DECLOP_1VAR_COMP(short1, !=)
DECLOP_1VAR_COMP(short1, <)
DECLOP_1VAR_COMP(short1, >)
DECLOP_1VAR_COMP(short1, <=)
DECLOP_1VAR_COMP(short1, >=)

DECLOP_1VAR_COMP(short1, &&)
DECLOP_1VAR_COMP(short1, ||)

DECLOP_1VAR_1IN_1OUT(short1, ~)
DECLOP_1VAR_1IN_BOOLOUT(short1, !)

DECLOP_1VAR_SCALE_PRODUCT(short1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(short1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(short1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(short1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(short1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(short1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(short1, float)
DECLOP_1VAR_SCALE_PRODUCT(short1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(short1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(short1, double)
DECLOP_1VAR_SCALE_PRODUCT(short1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(short1, signed long long)

// SIGNED SHORT2

DECLOP_2VAR_2IN_1OUT(short2, +)
DECLOP_2VAR_2IN_1OUT(short2, -)
DECLOP_2VAR_2IN_1OUT(short2, *)
DECLOP_2VAR_2IN_1OUT(short2, /)
DECLOP_2VAR_2IN_1OUT(short2, %)
DECLOP_2VAR_2IN_1OUT(short2, &)
DECLOP_2VAR_2IN_1OUT(short2, |)
DECLOP_2VAR_2IN_1OUT(short2, ^)
DECLOP_2VAR_2IN_1OUT(short2, <<)
DECLOP_2VAR_2IN_1OUT(short2, >>)

DECLOP_2VAR_ASSIGN(short2, +=)
DECLOP_2VAR_ASSIGN(short2, -=)
DECLOP_2VAR_ASSIGN(short2, *=)
DECLOP_2VAR_ASSIGN(short2, /=)
DECLOP_2VAR_ASSIGN(short2, %=)
DECLOP_2VAR_ASSIGN(short2, &=)
DECLOP_2VAR_ASSIGN(short2, |=)
DECLOP_2VAR_ASSIGN(short2, ^=)
DECLOP_2VAR_ASSIGN(short2, <<=)
DECLOP_2VAR_ASSIGN(short2, >>=)

DECLOP_2VAR_PREOP(short2, ++)
DECLOP_2VAR_PREOP(short2, --)

DECLOP_2VAR_POSTOP(short2, ++)
DECLOP_2VAR_POSTOP(short2, --)

DECLOP_2VAR_COMP(short2, ==)
DECLOP_2VAR_COMP(short2, !=)
DECLOP_2VAR_COMP(short2, <)
DECLOP_2VAR_COMP(short2, >)
DECLOP_2VAR_COMP(short2, <=)
DECLOP_2VAR_COMP(short2, >=)

DECLOP_2VAR_COMP(short2, &&)
DECLOP_2VAR_COMP(short2, ||)

DECLOP_2VAR_1IN_1OUT(short2, ~)
DECLOP_2VAR_1IN_BOOLOUT(short2, !)

DECLOP_2VAR_SCALE_PRODUCT(short2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(short2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(short2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(short2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(short2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(short2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(short2, float)
DECLOP_2VAR_SCALE_PRODUCT(short2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(short2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(short2, double)
DECLOP_2VAR_SCALE_PRODUCT(short2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(short2, signed long long)

// SIGNED SHORT3

DECLOP_3VAR_2IN_1OUT(short3, +)
DECLOP_3VAR_2IN_1OUT(short3, -)
DECLOP_3VAR_2IN_1OUT(short3, *)
DECLOP_3VAR_2IN_1OUT(short3, /)
DECLOP_3VAR_2IN_1OUT(short3, %)
DECLOP_3VAR_2IN_1OUT(short3, &)
DECLOP_3VAR_2IN_1OUT(short3, |)
DECLOP_3VAR_2IN_1OUT(short3, ^)
DECLOP_3VAR_2IN_1OUT(short3, <<)
DECLOP_3VAR_2IN_1OUT(short3, >>)

DECLOP_3VAR_ASSIGN(short3, +=)
DECLOP_3VAR_ASSIGN(short3, -=)
DECLOP_3VAR_ASSIGN(short3, *=)
DECLOP_3VAR_ASSIGN(short3, /=)
DECLOP_3VAR_ASSIGN(short3, %=)
DECLOP_3VAR_ASSIGN(short3, &=)
DECLOP_3VAR_ASSIGN(short3, |=)
DECLOP_3VAR_ASSIGN(short3, ^=)
DECLOP_3VAR_ASSIGN(short3, <<=)
DECLOP_3VAR_ASSIGN(short3, >>=)

DECLOP_3VAR_PREOP(short3, ++)
DECLOP_3VAR_PREOP(short3, --)

DECLOP_3VAR_POSTOP(short3, ++)
DECLOP_3VAR_POSTOP(short3, --)

DECLOP_3VAR_COMP(short3, ==)
DECLOP_3VAR_COMP(short3, !=)
DECLOP_3VAR_COMP(short3, <)
DECLOP_3VAR_COMP(short3, >)
DECLOP_3VAR_COMP(short3, <=)
DECLOP_3VAR_COMP(short3, >=)

DECLOP_3VAR_COMP(short3, &&)
DECLOP_3VAR_COMP(short3, ||)

DECLOP_3VAR_1IN_1OUT(short3, ~)
DECLOP_3VAR_1IN_BOOLOUT(short3, !)

DECLOP_3VAR_SCALE_PRODUCT(short3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(short3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(short3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(short3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(short3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(short3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(short3, float)
DECLOP_3VAR_SCALE_PRODUCT(short3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(short3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(short3, double)
DECLOP_3VAR_SCALE_PRODUCT(short3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(short3, signed long long)

// SIGNED SHORT4

DECLOP_4VAR_2IN_1OUT(short4, +)
DECLOP_4VAR_2IN_1OUT(short4, -)
DECLOP_4VAR_2IN_1OUT(short4, *)
DECLOP_4VAR_2IN_1OUT(short4, /)
DECLOP_4VAR_2IN_1OUT(short4, %)
DECLOP_4VAR_2IN_1OUT(short4, &)
DECLOP_4VAR_2IN_1OUT(short4, |)
DECLOP_4VAR_2IN_1OUT(short4, ^)
DECLOP_4VAR_2IN_1OUT(short4, <<)
DECLOP_4VAR_2IN_1OUT(short4, >>)

DECLOP_4VAR_ASSIGN(short4, +=)
DECLOP_4VAR_ASSIGN(short4, -=)
DECLOP_4VAR_ASSIGN(short4, *=)
DECLOP_4VAR_ASSIGN(short4, /=)
DECLOP_4VAR_ASSIGN(short4, %=)
DECLOP_4VAR_ASSIGN(short4, &=)
DECLOP_4VAR_ASSIGN(short4, |=)
DECLOP_4VAR_ASSIGN(short4, ^=)
DECLOP_4VAR_ASSIGN(short4, <<=)
DECLOP_4VAR_ASSIGN(short4, >>=)

DECLOP_4VAR_PREOP(short4, ++)
DECLOP_4VAR_PREOP(short4, --)

DECLOP_4VAR_POSTOP(short4, ++)
DECLOP_4VAR_POSTOP(short4, --)

DECLOP_4VAR_COMP(short4, ==)
DECLOP_4VAR_COMP(short4, !=)
DECLOP_4VAR_COMP(short4, <)
DECLOP_4VAR_COMP(short4, >)
DECLOP_4VAR_COMP(short4, <=)
DECLOP_4VAR_COMP(short4, >=)

DECLOP_4VAR_COMP(short4, &&)
DECLOP_4VAR_COMP(short4, ||)

DECLOP_4VAR_1IN_1OUT(short4, ~)
DECLOP_4VAR_1IN_BOOLOUT(short4, !)

DECLOP_4VAR_SCALE_PRODUCT(short4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(short4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(short4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(short4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(short4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(short4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(short4, float)
DECLOP_4VAR_SCALE_PRODUCT(short4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(short4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(short4, double)
DECLOP_4VAR_SCALE_PRODUCT(short4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(short4, signed long long)

// UNSIGNED INT1

DECLOP_1VAR_2IN_1OUT(uint1, +)
DECLOP_1VAR_2IN_1OUT(uint1, -)
DECLOP_1VAR_2IN_1OUT(uint1, *)
DECLOP_1VAR_2IN_1OUT(uint1, /)
DECLOP_1VAR_2IN_1OUT(uint1, %)
DECLOP_1VAR_2IN_1OUT(uint1, &)
DECLOP_1VAR_2IN_1OUT(uint1, |)
DECLOP_1VAR_2IN_1OUT(uint1, ^)
DECLOP_1VAR_2IN_1OUT(uint1, <<)
DECLOP_1VAR_2IN_1OUT(uint1, >>)


DECLOP_1VAR_ASSIGN(uint1, +=)
DECLOP_1VAR_ASSIGN(uint1, -=)
DECLOP_1VAR_ASSIGN(uint1, *=)
DECLOP_1VAR_ASSIGN(uint1, /=)
DECLOP_1VAR_ASSIGN(uint1, %=)
DECLOP_1VAR_ASSIGN(uint1, &=)
DECLOP_1VAR_ASSIGN(uint1, |=)
DECLOP_1VAR_ASSIGN(uint1, ^=)
DECLOP_1VAR_ASSIGN(uint1, <<=)
DECLOP_1VAR_ASSIGN(uint1, >>=)

DECLOP_1VAR_PREOP(uint1, ++)
DECLOP_1VAR_PREOP(uint1, --)

DECLOP_1VAR_POSTOP(uint1, ++)
DECLOP_1VAR_POSTOP(uint1, --)

DECLOP_1VAR_COMP(uint1, ==)
DECLOP_1VAR_COMP(uint1, !=)
DECLOP_1VAR_COMP(uint1, <)
DECLOP_1VAR_COMP(uint1, >)
DECLOP_1VAR_COMP(uint1, <=)
DECLOP_1VAR_COMP(uint1, >=)

DECLOP_1VAR_COMP(uint1, &&)
DECLOP_1VAR_COMP(uint1, ||)

DECLOP_1VAR_1IN_1OUT(uint1, ~)
DECLOP_1VAR_1IN_BOOLOUT(uint1, !)

DECLOP_1VAR_SCALE_PRODUCT(uint1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(uint1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(uint1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(uint1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(uint1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(uint1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(uint1, float)
DECLOP_1VAR_SCALE_PRODUCT(uint1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(uint1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(uint1, double)
DECLOP_1VAR_SCALE_PRODUCT(uint1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(uint1, signed long long)

// UNSIGNED INT2

DECLOP_2VAR_2IN_1OUT(uint2, +)
DECLOP_2VAR_2IN_1OUT(uint2, -)
DECLOP_2VAR_2IN_1OUT(uint2, *)
DECLOP_2VAR_2IN_1OUT(uint2, /)
DECLOP_2VAR_2IN_1OUT(uint2, %)
DECLOP_2VAR_2IN_1OUT(uint2, &)
DECLOP_2VAR_2IN_1OUT(uint2, |)
DECLOP_2VAR_2IN_1OUT(uint2, ^)
DECLOP_2VAR_2IN_1OUT(uint2, <<)
DECLOP_2VAR_2IN_1OUT(uint2, >>)

DECLOP_2VAR_ASSIGN(uint2, +=)
DECLOP_2VAR_ASSIGN(uint2, -=)
DECLOP_2VAR_ASSIGN(uint2, *=)
DECLOP_2VAR_ASSIGN(uint2, /=)
DECLOP_2VAR_ASSIGN(uint2, %=)
DECLOP_2VAR_ASSIGN(uint2, &=)
DECLOP_2VAR_ASSIGN(uint2, |=)
DECLOP_2VAR_ASSIGN(uint2, ^=)
DECLOP_2VAR_ASSIGN(uint2, <<=)
DECLOP_2VAR_ASSIGN(uint2, >>=)

DECLOP_2VAR_PREOP(uint2, ++)
DECLOP_2VAR_PREOP(uint2, --)

DECLOP_2VAR_POSTOP(uint2, ++)
DECLOP_2VAR_POSTOP(uint2, --)

DECLOP_2VAR_COMP(uint2, ==)
DECLOP_2VAR_COMP(uint2, !=)
DECLOP_2VAR_COMP(uint2, <)
DECLOP_2VAR_COMP(uint2, >)
DECLOP_2VAR_COMP(uint2, <=)
DECLOP_2VAR_COMP(uint2, >=)

DECLOP_2VAR_COMP(uint2, &&)
DECLOP_2VAR_COMP(uint2, ||)

DECLOP_2VAR_1IN_1OUT(uint2, ~)
DECLOP_2VAR_1IN_BOOLOUT(uint2, !)

DECLOP_2VAR_SCALE_PRODUCT(uint2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(uint2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(uint2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(uint2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(uint2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(uint2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(uint2, float)
DECLOP_2VAR_SCALE_PRODUCT(uint2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(uint2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(uint2, double)
DECLOP_2VAR_SCALE_PRODUCT(uint2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(uint2, signed long long)

// UNSIGNED INT3

DECLOP_3VAR_2IN_1OUT(uint3, +)
DECLOP_3VAR_2IN_1OUT(uint3, -)
DECLOP_3VAR_2IN_1OUT(uint3, *)
DECLOP_3VAR_2IN_1OUT(uint3, /)
DECLOP_3VAR_2IN_1OUT(uint3, %)
DECLOP_3VAR_2IN_1OUT(uint3, &)
DECLOP_3VAR_2IN_1OUT(uint3, |)
DECLOP_3VAR_2IN_1OUT(uint3, ^)
DECLOP_3VAR_2IN_1OUT(uint3, <<)
DECLOP_3VAR_2IN_1OUT(uint3, >>)

DECLOP_3VAR_ASSIGN(uint3, +=)
DECLOP_3VAR_ASSIGN(uint3, -=)
DECLOP_3VAR_ASSIGN(uint3, *=)
DECLOP_3VAR_ASSIGN(uint3, /=)
DECLOP_3VAR_ASSIGN(uint3, %=)
DECLOP_3VAR_ASSIGN(uint3, &=)
DECLOP_3VAR_ASSIGN(uint3, |=)
DECLOP_3VAR_ASSIGN(uint3, ^=)
DECLOP_3VAR_ASSIGN(uint3, <<=)
DECLOP_3VAR_ASSIGN(uint3, >>=)

DECLOP_3VAR_PREOP(uint3, ++)
DECLOP_3VAR_PREOP(uint3, --)

DECLOP_3VAR_POSTOP(uint3, ++)
DECLOP_3VAR_POSTOP(uint3, --)

DECLOP_3VAR_COMP(uint3, ==)
DECLOP_3VAR_COMP(uint3, !=)
DECLOP_3VAR_COMP(uint3, <)
DECLOP_3VAR_COMP(uint3, >)
DECLOP_3VAR_COMP(uint3, <=)
DECLOP_3VAR_COMP(uint3, >=)

DECLOP_3VAR_COMP(uint3, &&)
DECLOP_3VAR_COMP(uint3, ||)

DECLOP_3VAR_1IN_1OUT(uint3, ~)
DECLOP_3VAR_1IN_BOOLOUT(uint3, !)

DECLOP_3VAR_SCALE_PRODUCT(uint3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(uint3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(uint3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(uint3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(uint3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(uint3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(uint3, float)
DECLOP_3VAR_SCALE_PRODUCT(uint3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(uint3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(uint3, double)
DECLOP_3VAR_SCALE_PRODUCT(uint3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(uint3, signed long long)

// UNSIGNED INT4

DECLOP_4VAR_2IN_1OUT(uint4, +)
DECLOP_4VAR_2IN_1OUT(uint4, -)
DECLOP_4VAR_2IN_1OUT(uint4, *)
DECLOP_4VAR_2IN_1OUT(uint4, /)
DECLOP_4VAR_2IN_1OUT(uint4, %)
DECLOP_4VAR_2IN_1OUT(uint4, &)
DECLOP_4VAR_2IN_1OUT(uint4, |)
DECLOP_4VAR_2IN_1OUT(uint4, ^)
DECLOP_4VAR_2IN_1OUT(uint4, <<)
DECLOP_4VAR_2IN_1OUT(uint4, >>)

DECLOP_4VAR_ASSIGN(uint4, +=)
DECLOP_4VAR_ASSIGN(uint4, -=)
DECLOP_4VAR_ASSIGN(uint4, *=)
DECLOP_4VAR_ASSIGN(uint4, /=)
DECLOP_4VAR_ASSIGN(uint4, %=)
DECLOP_4VAR_ASSIGN(uint4, &=)
DECLOP_4VAR_ASSIGN(uint4, |=)
DECLOP_4VAR_ASSIGN(uint4, ^=)
DECLOP_4VAR_ASSIGN(uint4, <<=)
DECLOP_4VAR_ASSIGN(uint4, >>=)

DECLOP_4VAR_PREOP(uint4, ++)
DECLOP_4VAR_PREOP(uint4, --)

DECLOP_4VAR_POSTOP(uint4, ++)
DECLOP_4VAR_POSTOP(uint4, --)

DECLOP_4VAR_COMP(uint4, ==)
DECLOP_4VAR_COMP(uint4, !=)
DECLOP_4VAR_COMP(uint4, <)
DECLOP_4VAR_COMP(uint4, >)
DECLOP_4VAR_COMP(uint4, <=)
DECLOP_4VAR_COMP(uint4, >=)

DECLOP_4VAR_COMP(uint4, &&)
DECLOP_4VAR_COMP(uint4, ||)

DECLOP_4VAR_1IN_1OUT(uint4, ~)
DECLOP_4VAR_1IN_BOOLOUT(uint4, !)

DECLOP_4VAR_SCALE_PRODUCT(uint4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(uint4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(uint4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(uint4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(uint4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(uint4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(uint4, float)
DECLOP_4VAR_SCALE_PRODUCT(uint4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(uint4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(uint4, double)
DECLOP_4VAR_SCALE_PRODUCT(uint4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(uint4, signed long long)

// SIGNED INT1

DECLOP_1VAR_2IN_1OUT(int1, +)
DECLOP_1VAR_2IN_1OUT(int1, -)
DECLOP_1VAR_2IN_1OUT(int1, *)
DECLOP_1VAR_2IN_1OUT(int1, /)
DECLOP_1VAR_2IN_1OUT(int1, %)
DECLOP_1VAR_2IN_1OUT(int1, &)
DECLOP_1VAR_2IN_1OUT(int1, |)
DECLOP_1VAR_2IN_1OUT(int1, ^)
DECLOP_1VAR_2IN_1OUT(int1, <<)
DECLOP_1VAR_2IN_1OUT(int1, >>)


DECLOP_1VAR_ASSIGN(int1, +=)
DECLOP_1VAR_ASSIGN(int1, -=)
DECLOP_1VAR_ASSIGN(int1, *=)
DECLOP_1VAR_ASSIGN(int1, /=)
DECLOP_1VAR_ASSIGN(int1, %=)
DECLOP_1VAR_ASSIGN(int1, &=)
DECLOP_1VAR_ASSIGN(int1, |=)
DECLOP_1VAR_ASSIGN(int1, ^=)
DECLOP_1VAR_ASSIGN(int1, <<=)
DECLOP_1VAR_ASSIGN(int1, >>=)

DECLOP_1VAR_PREOP(int1, ++)
DECLOP_1VAR_PREOP(int1, --)

DECLOP_1VAR_POSTOP(int1, ++)
DECLOP_1VAR_POSTOP(int1, --)

DECLOP_1VAR_COMP(int1, ==)
DECLOP_1VAR_COMP(int1, !=)
DECLOP_1VAR_COMP(int1, <)
DECLOP_1VAR_COMP(int1, >)
DECLOP_1VAR_COMP(int1, <=)
DECLOP_1VAR_COMP(int1, >=)

DECLOP_1VAR_COMP(int1, &&)
DECLOP_1VAR_COMP(int1, ||)

DECLOP_1VAR_1IN_1OUT(int1, ~)
DECLOP_1VAR_1IN_BOOLOUT(int1, !)

DECLOP_1VAR_SCALE_PRODUCT(int1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(int1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(int1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(int1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(int1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(int1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(int1, float)
DECLOP_1VAR_SCALE_PRODUCT(int1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(int1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(int1, double)
DECLOP_1VAR_SCALE_PRODUCT(int1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(int1, signed long long)

// SIGNED INT2

DECLOP_2VAR_2IN_1OUT(int2, +)
DECLOP_2VAR_2IN_1OUT(int2, -)
DECLOP_2VAR_2IN_1OUT(int2, *)
DECLOP_2VAR_2IN_1OUT(int2, /)
DECLOP_2VAR_2IN_1OUT(int2, %)
DECLOP_2VAR_2IN_1OUT(int2, &)
DECLOP_2VAR_2IN_1OUT(int2, |)
DECLOP_2VAR_2IN_1OUT(int2, ^)
DECLOP_2VAR_2IN_1OUT(int2, <<)
DECLOP_2VAR_2IN_1OUT(int2, >>)

DECLOP_2VAR_ASSIGN(int2, +=)
DECLOP_2VAR_ASSIGN(int2, -=)
DECLOP_2VAR_ASSIGN(int2, *=)
DECLOP_2VAR_ASSIGN(int2, /=)
DECLOP_2VAR_ASSIGN(int2, %=)
DECLOP_2VAR_ASSIGN(int2, &=)
DECLOP_2VAR_ASSIGN(int2, |=)
DECLOP_2VAR_ASSIGN(int2, ^=)
DECLOP_2VAR_ASSIGN(int2, <<=)
DECLOP_2VAR_ASSIGN(int2, >>=)

DECLOP_2VAR_PREOP(int2, ++)
DECLOP_2VAR_PREOP(int2, --)

DECLOP_2VAR_POSTOP(int2, ++)
DECLOP_2VAR_POSTOP(int2, --)

DECLOP_2VAR_COMP(int2, ==)
DECLOP_2VAR_COMP(int2, !=)
DECLOP_2VAR_COMP(int2, <)
DECLOP_2VAR_COMP(int2, >)
DECLOP_2VAR_COMP(int2, <=)
DECLOP_2VAR_COMP(int2, >=)

DECLOP_2VAR_COMP(int2, &&)
DECLOP_2VAR_COMP(int2, ||)

DECLOP_2VAR_1IN_1OUT(int2, ~)
DECLOP_2VAR_1IN_BOOLOUT(int2, !)

DECLOP_2VAR_SCALE_PRODUCT(int2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(int2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(int2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(int2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(int2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(int2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(int2, float)
DECLOP_2VAR_SCALE_PRODUCT(int2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(int2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(int2, double)
DECLOP_2VAR_SCALE_PRODUCT(int2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(int2, signed long long)

// SIGNED INT3

DECLOP_3VAR_2IN_1OUT(int3, +)
DECLOP_3VAR_2IN_1OUT(int3, -)
DECLOP_3VAR_2IN_1OUT(int3, *)
DECLOP_3VAR_2IN_1OUT(int3, /)
DECLOP_3VAR_2IN_1OUT(int3, %)
DECLOP_3VAR_2IN_1OUT(int3, &)
DECLOP_3VAR_2IN_1OUT(int3, |)
DECLOP_3VAR_2IN_1OUT(int3, ^)
DECLOP_3VAR_2IN_1OUT(int3, <<)
DECLOP_3VAR_2IN_1OUT(int3, >>)

DECLOP_3VAR_ASSIGN(int3, +=)
DECLOP_3VAR_ASSIGN(int3, -=)
DECLOP_3VAR_ASSIGN(int3, *=)
DECLOP_3VAR_ASSIGN(int3, /=)
DECLOP_3VAR_ASSIGN(int3, %=)
DECLOP_3VAR_ASSIGN(int3, &=)
DECLOP_3VAR_ASSIGN(int3, |=)
DECLOP_3VAR_ASSIGN(int3, ^=)
DECLOP_3VAR_ASSIGN(int3, <<=)
DECLOP_3VAR_ASSIGN(int3, >>=)

DECLOP_3VAR_PREOP(int3, ++)
DECLOP_3VAR_PREOP(int3, --)

DECLOP_3VAR_POSTOP(int3, ++)
DECLOP_3VAR_POSTOP(int3, --)

DECLOP_3VAR_COMP(int3, ==)
DECLOP_3VAR_COMP(int3, !=)
DECLOP_3VAR_COMP(int3, <)
DECLOP_3VAR_COMP(int3, >)
DECLOP_3VAR_COMP(int3, <=)
DECLOP_3VAR_COMP(int3, >=)

DECLOP_3VAR_COMP(int3, &&)
DECLOP_3VAR_COMP(int3, ||)

DECLOP_3VAR_1IN_1OUT(int3, ~)
DECLOP_3VAR_1IN_BOOLOUT(int3, !)

DECLOP_3VAR_SCALE_PRODUCT(int3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(int3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(int3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(int3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(int3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(int3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(int3, float)
DECLOP_3VAR_SCALE_PRODUCT(int3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(int3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(int3, double)
DECLOP_3VAR_SCALE_PRODUCT(int3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(int3, signed long long)

// SIGNED INT4

DECLOP_4VAR_2IN_1OUT(int4, +)
DECLOP_4VAR_2IN_1OUT(int4, -)
DECLOP_4VAR_2IN_1OUT(int4, *)
DECLOP_4VAR_2IN_1OUT(int4, /)
DECLOP_4VAR_2IN_1OUT(int4, %)
DECLOP_4VAR_2IN_1OUT(int4, &)
DECLOP_4VAR_2IN_1OUT(int4, |)
DECLOP_4VAR_2IN_1OUT(int4, ^)
DECLOP_4VAR_2IN_1OUT(int4, <<)
DECLOP_4VAR_2IN_1OUT(int4, >>)

DECLOP_4VAR_ASSIGN(int4, +=)
DECLOP_4VAR_ASSIGN(int4, -=)
DECLOP_4VAR_ASSIGN(int4, *=)
DECLOP_4VAR_ASSIGN(int4, /=)
DECLOP_4VAR_ASSIGN(int4, %=)
DECLOP_4VAR_ASSIGN(int4, &=)
DECLOP_4VAR_ASSIGN(int4, |=)
DECLOP_4VAR_ASSIGN(int4, ^=)
DECLOP_4VAR_ASSIGN(int4, <<=)
DECLOP_4VAR_ASSIGN(int4, >>=)

DECLOP_4VAR_PREOP(int4, ++)
DECLOP_4VAR_PREOP(int4, --)

DECLOP_4VAR_POSTOP(int4, ++)
DECLOP_4VAR_POSTOP(int4, --)

DECLOP_4VAR_COMP(int4, ==)
DECLOP_4VAR_COMP(int4, !=)
DECLOP_4VAR_COMP(int4, <)
DECLOP_4VAR_COMP(int4, >)
DECLOP_4VAR_COMP(int4, <=)
DECLOP_4VAR_COMP(int4, >=)

DECLOP_4VAR_COMP(int4, &&)
DECLOP_4VAR_COMP(int4, ||)

DECLOP_4VAR_1IN_1OUT(int4, ~)
DECLOP_4VAR_1IN_BOOLOUT(int4, !)

DECLOP_4VAR_SCALE_PRODUCT(int4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(int4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(int4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(int4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(int4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(int4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(int4, float)
DECLOP_4VAR_SCALE_PRODUCT(int4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(int4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(int4, double)
DECLOP_4VAR_SCALE_PRODUCT(int4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(int4, signed long long)

// FLOAT1

DECLOP_1VAR_2IN_1OUT(float1, +)
DECLOP_1VAR_2IN_1OUT(float1, -)
DECLOP_1VAR_2IN_1OUT(float1, *)
DECLOP_1VAR_2IN_1OUT(float1, /)

DECLOP_1VAR_ASSIGN(float1, +=)
DECLOP_1VAR_ASSIGN(float1, -=)
DECLOP_1VAR_ASSIGN(float1, *=)
DECLOP_1VAR_ASSIGN(float1, /=)

DECLOP_1VAR_PREOP(float1, ++)
DECLOP_1VAR_PREOP(float1, --)

DECLOP_1VAR_POSTOP(float1, ++)
DECLOP_1VAR_POSTOP(float1, --)

DECLOP_1VAR_COMP(float1, ==)
DECLOP_1VAR_COMP(float1, !=)
DECLOP_1VAR_COMP(float1, <)
DECLOP_1VAR_COMP(float1, >)
DECLOP_1VAR_COMP(float1, <=)
DECLOP_1VAR_COMP(float1, >=)

DECLOP_1VAR_SCALE_PRODUCT(float1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(float1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(float1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(float1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(float1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(float1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(float1, float)
DECLOP_1VAR_SCALE_PRODUCT(float1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(float1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(float1, double)
DECLOP_1VAR_SCALE_PRODUCT(float1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(float1, signed long long)

// FLOAT2

DECLOP_2VAR_2IN_1OUT(float2, +)
DECLOP_2VAR_2IN_1OUT(float2, -)
DECLOP_2VAR_2IN_1OUT(float2, *)
DECLOP_2VAR_2IN_1OUT(float2, /)

DECLOP_2VAR_ASSIGN(float2, +=)
DECLOP_2VAR_ASSIGN(float2, -=)
DECLOP_2VAR_ASSIGN(float2, *=)
DECLOP_2VAR_ASSIGN(float2, /=)

DECLOP_2VAR_PREOP(float2, ++)
DECLOP_2VAR_PREOP(float2, --)

DECLOP_2VAR_POSTOP(float2, ++)
DECLOP_2VAR_POSTOP(float2, --)

DECLOP_2VAR_COMP(float2, ==)
DECLOP_2VAR_COMP(float2, !=)
DECLOP_2VAR_COMP(float2, <)
DECLOP_2VAR_COMP(float2, >)
DECLOP_2VAR_COMP(float2, <=)
DECLOP_2VAR_COMP(float2, >=)

DECLOP_2VAR_SCALE_PRODUCT(float2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(float2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(float2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(float2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(float2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(float2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(float2, float)
DECLOP_2VAR_SCALE_PRODUCT(float2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(float2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(float2, double)
DECLOP_2VAR_SCALE_PRODUCT(float2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(float2, signed long long)

// FLOAT3

DECLOP_3VAR_2IN_1OUT(float3, +)
DECLOP_3VAR_2IN_1OUT(float3, -)
DECLOP_3VAR_2IN_1OUT(float3, *)
DECLOP_3VAR_2IN_1OUT(float3, /)

DECLOP_3VAR_ASSIGN(float3, +=)
DECLOP_3VAR_ASSIGN(float3, -=)
DECLOP_3VAR_ASSIGN(float3, *=)
DECLOP_3VAR_ASSIGN(float3, /=)

DECLOP_3VAR_PREOP(float3, ++)
DECLOP_3VAR_PREOP(float3, --)

DECLOP_3VAR_POSTOP(float3, ++)
DECLOP_3VAR_POSTOP(float3, --)

DECLOP_3VAR_COMP(float3, ==)
DECLOP_3VAR_COMP(float3, !=)
DECLOP_3VAR_COMP(float3, <)
DECLOP_3VAR_COMP(float3, >)
DECLOP_3VAR_COMP(float3, <=)
DECLOP_3VAR_COMP(float3, >=)

DECLOP_3VAR_SCALE_PRODUCT(float3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(float3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(float3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(float3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(float3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(float3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(float3, float)
DECLOP_3VAR_SCALE_PRODUCT(float3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(float3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(float3, double)
DECLOP_3VAR_SCALE_PRODUCT(float3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(float3, signed long long)

// FLOAT4

DECLOP_4VAR_2IN_1OUT(float4, +)
DECLOP_4VAR_2IN_1OUT(float4, -)
DECLOP_4VAR_2IN_1OUT(float4, *)
DECLOP_4VAR_2IN_1OUT(float4, /)

DECLOP_4VAR_ASSIGN(float4, +=)
DECLOP_4VAR_ASSIGN(float4, -=)
DECLOP_4VAR_ASSIGN(float4, *=)
DECLOP_4VAR_ASSIGN(float4, /=)

DECLOP_4VAR_PREOP(float4, ++)
DECLOP_4VAR_PREOP(float4, --)

DECLOP_4VAR_POSTOP(float4, ++)
DECLOP_4VAR_POSTOP(float4, --)

DECLOP_4VAR_COMP(float4, ==)
DECLOP_4VAR_COMP(float4, !=)
DECLOP_4VAR_COMP(float4, <)
DECLOP_4VAR_COMP(float4, >)
DECLOP_4VAR_COMP(float4, <=)
DECLOP_4VAR_COMP(float4, >=)

DECLOP_4VAR_SCALE_PRODUCT(float4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(float4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(float4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(float4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(float4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(float4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(float4, float)
DECLOP_4VAR_SCALE_PRODUCT(float4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(float4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(float4, double)
DECLOP_4VAR_SCALE_PRODUCT(float4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(float4, signed long long)

// DOUBLE1

DECLOP_1VAR_2IN_1OUT(double1, +)
DECLOP_1VAR_2IN_1OUT(double1, -)
DECLOP_1VAR_2IN_1OUT(double1, *)
DECLOP_1VAR_2IN_1OUT(double1, /)

DECLOP_1VAR_ASSIGN(double1, +=)
DECLOP_1VAR_ASSIGN(double1, -=)
DECLOP_1VAR_ASSIGN(double1, *=)
DECLOP_1VAR_ASSIGN(double1, /=)

DECLOP_1VAR_PREOP(double1, ++)
DECLOP_1VAR_PREOP(double1, --)

DECLOP_1VAR_POSTOP(double1, ++)
DECLOP_1VAR_POSTOP(double1, --)

DECLOP_1VAR_COMP(double1, ==)
DECLOP_1VAR_COMP(double1, !=)
DECLOP_1VAR_COMP(double1, <)
DECLOP_1VAR_COMP(double1, >)
DECLOP_1VAR_COMP(double1, <=)
DECLOP_1VAR_COMP(double1, >=)

DECLOP_1VAR_SCALE_PRODUCT(double1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(double1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(double1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(double1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(double1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(double1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(double1, float)
DECLOP_1VAR_SCALE_PRODUCT(double1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(double1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(double1, double)
DECLOP_1VAR_SCALE_PRODUCT(double1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(double1, signed long long)

// DOUBLE2

DECLOP_2VAR_2IN_1OUT(double2, +)
DECLOP_2VAR_2IN_1OUT(double2, -)
DECLOP_2VAR_2IN_1OUT(double2, *)
DECLOP_2VAR_2IN_1OUT(double2, /)

DECLOP_2VAR_ASSIGN(double2, +=)
DECLOP_2VAR_ASSIGN(double2, -=)
DECLOP_2VAR_ASSIGN(double2, *=)
DECLOP_2VAR_ASSIGN(double2, /=)

DECLOP_2VAR_PREOP(double2, ++)
DECLOP_2VAR_PREOP(double2, --)

DECLOP_2VAR_POSTOP(double2, ++)
DECLOP_2VAR_POSTOP(double2, --)

DECLOP_2VAR_COMP(double2, ==)
DECLOP_2VAR_COMP(double2, !=)
DECLOP_2VAR_COMP(double2, <)
DECLOP_2VAR_COMP(double2, >)
DECLOP_2VAR_COMP(double2, <=)
DECLOP_2VAR_COMP(double2, >=)

DECLOP_2VAR_SCALE_PRODUCT(double2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(double2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(double2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(double2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(double2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(double2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(double2, float)
DECLOP_2VAR_SCALE_PRODUCT(double2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(double2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(double2, double)
DECLOP_2VAR_SCALE_PRODUCT(double2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(double2, signed long long)

// DOUBLE3

DECLOP_3VAR_2IN_1OUT(double3, +)
DECLOP_3VAR_2IN_1OUT(double3, -)
DECLOP_3VAR_2IN_1OUT(double3, *)
DECLOP_3VAR_2IN_1OUT(double3, /)

DECLOP_3VAR_ASSIGN(double3, +=)
DECLOP_3VAR_ASSIGN(double3, -=)
DECLOP_3VAR_ASSIGN(double3, *=)
DECLOP_3VAR_ASSIGN(double3, /=)

DECLOP_3VAR_PREOP(double3, ++)
DECLOP_3VAR_PREOP(double3, --)

DECLOP_3VAR_POSTOP(double3, ++)
DECLOP_3VAR_POSTOP(double3, --)

DECLOP_3VAR_COMP(double3, ==)
DECLOP_3VAR_COMP(double3, !=)
DECLOP_3VAR_COMP(double3, <)
DECLOP_3VAR_COMP(double3, >)
DECLOP_3VAR_COMP(double3, <=)
DECLOP_3VAR_COMP(double3, >=)

DECLOP_3VAR_SCALE_PRODUCT(double3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(double3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(double3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(double3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(double3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(double3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(double3, float)
DECLOP_3VAR_SCALE_PRODUCT(double3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(double3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(double3, double)
DECLOP_3VAR_SCALE_PRODUCT(double3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(double3, signed long long)

// DOUBLE4

DECLOP_4VAR_2IN_1OUT(double4, +)
DECLOP_4VAR_2IN_1OUT(double4, -)
DECLOP_4VAR_2IN_1OUT(double4, *)
DECLOP_4VAR_2IN_1OUT(double4, /)

DECLOP_4VAR_ASSIGN(double4, +=)
DECLOP_4VAR_ASSIGN(double4, -=)
DECLOP_4VAR_ASSIGN(double4, *=)
DECLOP_4VAR_ASSIGN(double4, /=)

DECLOP_4VAR_PREOP(double4, ++)
DECLOP_4VAR_PREOP(double4, --)

DECLOP_4VAR_POSTOP(double4, ++)
DECLOP_4VAR_POSTOP(double4, --)

DECLOP_4VAR_COMP(double4, ==)
DECLOP_4VAR_COMP(double4, !=)
DECLOP_4VAR_COMP(double4, <)
DECLOP_4VAR_COMP(double4, >)
DECLOP_4VAR_COMP(double4, <=)
DECLOP_4VAR_COMP(double4, >=)

DECLOP_4VAR_SCALE_PRODUCT(double4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(double4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(double4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(double4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(double4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(double4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(double4, float)
DECLOP_4VAR_SCALE_PRODUCT(double4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(double4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(double4, double)
DECLOP_4VAR_SCALE_PRODUCT(double4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(double4, signed long long)

// UNSIGNED LONG1

DECLOP_1VAR_2IN_1OUT(ulong1, +)
DECLOP_1VAR_2IN_1OUT(ulong1, -)
DECLOP_1VAR_2IN_1OUT(ulong1, *)
DECLOP_1VAR_2IN_1OUT(ulong1, /)
DECLOP_1VAR_2IN_1OUT(ulong1, %)
DECLOP_1VAR_2IN_1OUT(ulong1, &)
DECLOP_1VAR_2IN_1OUT(ulong1, |)
DECLOP_1VAR_2IN_1OUT(ulong1, ^)
DECLOP_1VAR_2IN_1OUT(ulong1, <<)
DECLOP_1VAR_2IN_1OUT(ulong1, >>)


DECLOP_1VAR_ASSIGN(ulong1, +=)
DECLOP_1VAR_ASSIGN(ulong1, -=)
DECLOP_1VAR_ASSIGN(ulong1, *=)
DECLOP_1VAR_ASSIGN(ulong1, /=)
DECLOP_1VAR_ASSIGN(ulong1, %=)
DECLOP_1VAR_ASSIGN(ulong1, &=)
DECLOP_1VAR_ASSIGN(ulong1, |=)
DECLOP_1VAR_ASSIGN(ulong1, ^=)
DECLOP_1VAR_ASSIGN(ulong1, <<=)
DECLOP_1VAR_ASSIGN(ulong1, >>=)

DECLOP_1VAR_PREOP(ulong1, ++)
DECLOP_1VAR_PREOP(ulong1, --)

DECLOP_1VAR_POSTOP(ulong1, ++)
DECLOP_1VAR_POSTOP(ulong1, --)

DECLOP_1VAR_COMP(ulong1, ==)
DECLOP_1VAR_COMP(ulong1, !=)
DECLOP_1VAR_COMP(ulong1, <)
DECLOP_1VAR_COMP(ulong1, >)
DECLOP_1VAR_COMP(ulong1, <=)
DECLOP_1VAR_COMP(ulong1, >=)

DECLOP_1VAR_COMP(ulong1, &&)
DECLOP_1VAR_COMP(ulong1, ||)

DECLOP_1VAR_1IN_1OUT(ulong1, ~)
DECLOP_1VAR_1IN_BOOLOUT(ulong1, !)

DECLOP_1VAR_SCALE_PRODUCT(ulong1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, float)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, double)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(ulong1, signed long long)

// UNSIGNED LONG2

DECLOP_2VAR_2IN_1OUT(ulong2, +)
DECLOP_2VAR_2IN_1OUT(ulong2, -)
DECLOP_2VAR_2IN_1OUT(ulong2, *)
DECLOP_2VAR_2IN_1OUT(ulong2, /)
DECLOP_2VAR_2IN_1OUT(ulong2, %)
DECLOP_2VAR_2IN_1OUT(ulong2, &)
DECLOP_2VAR_2IN_1OUT(ulong2, |)
DECLOP_2VAR_2IN_1OUT(ulong2, ^)
DECLOP_2VAR_2IN_1OUT(ulong2, <<)
DECLOP_2VAR_2IN_1OUT(ulong2, >>)

DECLOP_2VAR_ASSIGN(ulong2, +=)
DECLOP_2VAR_ASSIGN(ulong2, -=)
DECLOP_2VAR_ASSIGN(ulong2, *=)
DECLOP_2VAR_ASSIGN(ulong2, /=)
DECLOP_2VAR_ASSIGN(ulong2, %=)
DECLOP_2VAR_ASSIGN(ulong2, &=)
DECLOP_2VAR_ASSIGN(ulong2, |=)
DECLOP_2VAR_ASSIGN(ulong2, ^=)
DECLOP_2VAR_ASSIGN(ulong2, <<=)
DECLOP_2VAR_ASSIGN(ulong2, >>=)

DECLOP_2VAR_PREOP(ulong2, ++)
DECLOP_2VAR_PREOP(ulong2, --)

DECLOP_2VAR_POSTOP(ulong2, ++)
DECLOP_2VAR_POSTOP(ulong2, --)

DECLOP_2VAR_COMP(ulong2, ==)
DECLOP_2VAR_COMP(ulong2, !=)
DECLOP_2VAR_COMP(ulong2, <)
DECLOP_2VAR_COMP(ulong2, >)
DECLOP_2VAR_COMP(ulong2, <=)
DECLOP_2VAR_COMP(ulong2, >=)

DECLOP_2VAR_COMP(ulong2, &&)
DECLOP_2VAR_COMP(ulong2, ||)

DECLOP_2VAR_1IN_1OUT(ulong2, ~)
DECLOP_2VAR_1IN_BOOLOUT(ulong2, !)

DECLOP_2VAR_SCALE_PRODUCT(ulong2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, float)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, double)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(ulong2, signed long long)

// UNSIGNED LONG3

DECLOP_3VAR_2IN_1OUT(ulong3, +)
DECLOP_3VAR_2IN_1OUT(ulong3, -)
DECLOP_3VAR_2IN_1OUT(ulong3, *)
DECLOP_3VAR_2IN_1OUT(ulong3, /)
DECLOP_3VAR_2IN_1OUT(ulong3, %)
DECLOP_3VAR_2IN_1OUT(ulong3, &)
DECLOP_3VAR_2IN_1OUT(ulong3, |)
DECLOP_3VAR_2IN_1OUT(ulong3, ^)
DECLOP_3VAR_2IN_1OUT(ulong3, <<)
DECLOP_3VAR_2IN_1OUT(ulong3, >>)

DECLOP_3VAR_ASSIGN(ulong3, +=)
DECLOP_3VAR_ASSIGN(ulong3, -=)
DECLOP_3VAR_ASSIGN(ulong3, *=)
DECLOP_3VAR_ASSIGN(ulong3, /=)
DECLOP_3VAR_ASSIGN(ulong3, %=)
DECLOP_3VAR_ASSIGN(ulong3, &=)
DECLOP_3VAR_ASSIGN(ulong3, |=)
DECLOP_3VAR_ASSIGN(ulong3, ^=)
DECLOP_3VAR_ASSIGN(ulong3, <<=)
DECLOP_3VAR_ASSIGN(ulong3, >>=)

DECLOP_3VAR_PREOP(ulong3, ++)
DECLOP_3VAR_PREOP(ulong3, --)

DECLOP_3VAR_POSTOP(ulong3, ++)
DECLOP_3VAR_POSTOP(ulong3, --)

DECLOP_3VAR_COMP(ulong3, ==)
DECLOP_3VAR_COMP(ulong3, !=)
DECLOP_3VAR_COMP(ulong3, <)
DECLOP_3VAR_COMP(ulong3, >)
DECLOP_3VAR_COMP(ulong3, <=)
DECLOP_3VAR_COMP(ulong3, >=)

DECLOP_3VAR_COMP(ulong3, &&)
DECLOP_3VAR_COMP(ulong3, ||)

DECLOP_3VAR_1IN_1OUT(ulong3, ~)
DECLOP_3VAR_1IN_BOOLOUT(ulong3, !)

DECLOP_3VAR_SCALE_PRODUCT(ulong3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, float)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, double)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(ulong3, signed long long)

// UNSIGNED LONG4

DECLOP_4VAR_2IN_1OUT(ulong4, +)
DECLOP_4VAR_2IN_1OUT(ulong4, -)
DECLOP_4VAR_2IN_1OUT(ulong4, *)
DECLOP_4VAR_2IN_1OUT(ulong4, /)
DECLOP_4VAR_2IN_1OUT(ulong4, %)
DECLOP_4VAR_2IN_1OUT(ulong4, &)
DECLOP_4VAR_2IN_1OUT(ulong4, |)
DECLOP_4VAR_2IN_1OUT(ulong4, ^)
DECLOP_4VAR_2IN_1OUT(ulong4, <<)
DECLOP_4VAR_2IN_1OUT(ulong4, >>)

DECLOP_4VAR_ASSIGN(ulong4, +=)
DECLOP_4VAR_ASSIGN(ulong4, -=)
DECLOP_4VAR_ASSIGN(ulong4, *=)
DECLOP_4VAR_ASSIGN(ulong4, /=)
DECLOP_4VAR_ASSIGN(ulong4, %=)
DECLOP_4VAR_ASSIGN(ulong4, &=)
DECLOP_4VAR_ASSIGN(ulong4, |=)
DECLOP_4VAR_ASSIGN(ulong4, ^=)
DECLOP_4VAR_ASSIGN(ulong4, <<=)
DECLOP_4VAR_ASSIGN(ulong4, >>=)

DECLOP_4VAR_PREOP(ulong4, ++)
DECLOP_4VAR_PREOP(ulong4, --)

DECLOP_4VAR_POSTOP(ulong4, ++)
DECLOP_4VAR_POSTOP(ulong4, --)

DECLOP_4VAR_COMP(ulong4, ==)
DECLOP_4VAR_COMP(ulong4, !=)
DECLOP_4VAR_COMP(ulong4, <)
DECLOP_4VAR_COMP(ulong4, >)
DECLOP_4VAR_COMP(ulong4, <=)
DECLOP_4VAR_COMP(ulong4, >=)

DECLOP_4VAR_COMP(ulong4, &&)
DECLOP_4VAR_COMP(ulong4, ||)

DECLOP_4VAR_1IN_1OUT(ulong4, ~)
DECLOP_4VAR_1IN_BOOLOUT(ulong4, !)

DECLOP_4VAR_SCALE_PRODUCT(ulong4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, float)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, double)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(ulong4, signed long long)

// SIGNED LONG1

DECLOP_1VAR_2IN_1OUT(long1, +)
DECLOP_1VAR_2IN_1OUT(long1, -)
DECLOP_1VAR_2IN_1OUT(long1, *)
DECLOP_1VAR_2IN_1OUT(long1, /)
DECLOP_1VAR_2IN_1OUT(long1, %)
DECLOP_1VAR_2IN_1OUT(long1, &)
DECLOP_1VAR_2IN_1OUT(long1, |)
DECLOP_1VAR_2IN_1OUT(long1, ^)
DECLOP_1VAR_2IN_1OUT(long1, <<)
DECLOP_1VAR_2IN_1OUT(long1, >>)


DECLOP_1VAR_ASSIGN(long1, +=)
DECLOP_1VAR_ASSIGN(long1, -=)
DECLOP_1VAR_ASSIGN(long1, *=)
DECLOP_1VAR_ASSIGN(long1, /=)
DECLOP_1VAR_ASSIGN(long1, %=)
DECLOP_1VAR_ASSIGN(long1, &=)
DECLOP_1VAR_ASSIGN(long1, |=)
DECLOP_1VAR_ASSIGN(long1, ^=)
DECLOP_1VAR_ASSIGN(long1, <<=)
DECLOP_1VAR_ASSIGN(long1, >>=)

DECLOP_1VAR_PREOP(long1, ++)
DECLOP_1VAR_PREOP(long1, --)

DECLOP_1VAR_POSTOP(long1, ++)
DECLOP_1VAR_POSTOP(long1, --)

DECLOP_1VAR_COMP(long1, ==)
DECLOP_1VAR_COMP(long1, !=)
DECLOP_1VAR_COMP(long1, <)
DECLOP_1VAR_COMP(long1, >)
DECLOP_1VAR_COMP(long1, <=)
DECLOP_1VAR_COMP(long1, >=)

DECLOP_1VAR_COMP(long1, &&)
DECLOP_1VAR_COMP(long1, ||)

DECLOP_1VAR_1IN_1OUT(long1, ~)
DECLOP_1VAR_1IN_BOOLOUT(long1, !)

DECLOP_1VAR_SCALE_PRODUCT(long1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(long1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(long1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(long1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(long1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(long1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(long1, float)
DECLOP_1VAR_SCALE_PRODUCT(long1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(long1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(long1, double)
DECLOP_1VAR_SCALE_PRODUCT(long1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(long1, signed long long)

// SIGNED LONG2

DECLOP_2VAR_2IN_1OUT(long2, +)
DECLOP_2VAR_2IN_1OUT(long2, -)
DECLOP_2VAR_2IN_1OUT(long2, *)
DECLOP_2VAR_2IN_1OUT(long2, /)
DECLOP_2VAR_2IN_1OUT(long2, %)
DECLOP_2VAR_2IN_1OUT(long2, &)
DECLOP_2VAR_2IN_1OUT(long2, |)
DECLOP_2VAR_2IN_1OUT(long2, ^)
DECLOP_2VAR_2IN_1OUT(long2, <<)
DECLOP_2VAR_2IN_1OUT(long2, >>)

DECLOP_2VAR_ASSIGN(long2, +=)
DECLOP_2VAR_ASSIGN(long2, -=)
DECLOP_2VAR_ASSIGN(long2, *=)
DECLOP_2VAR_ASSIGN(long2, /=)
DECLOP_2VAR_ASSIGN(long2, %=)
DECLOP_2VAR_ASSIGN(long2, &=)
DECLOP_2VAR_ASSIGN(long2, |=)
DECLOP_2VAR_ASSIGN(long2, ^=)
DECLOP_2VAR_ASSIGN(long2, <<=)
DECLOP_2VAR_ASSIGN(long2, >>=)

DECLOP_2VAR_PREOP(long2, ++)
DECLOP_2VAR_PREOP(long2, --)

DECLOP_2VAR_POSTOP(long2, ++)
DECLOP_2VAR_POSTOP(long2, --)

DECLOP_2VAR_COMP(long2, ==)
DECLOP_2VAR_COMP(long2, !=)
DECLOP_2VAR_COMP(long2, <)
DECLOP_2VAR_COMP(long2, >)
DECLOP_2VAR_COMP(long2, <=)
DECLOP_2VAR_COMP(long2, >=)

DECLOP_2VAR_COMP(long2, &&)
DECLOP_2VAR_COMP(long2, ||)

DECLOP_2VAR_1IN_1OUT(long2, ~)
DECLOP_2VAR_1IN_BOOLOUT(long2, !)

DECLOP_2VAR_SCALE_PRODUCT(long2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(long2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(long2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(long2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(long2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(long2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(long2, float)
DECLOP_2VAR_SCALE_PRODUCT(long2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(long2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(long2, double)
DECLOP_2VAR_SCALE_PRODUCT(long2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(long2, signed long long)

// SIGNED LONG3

DECLOP_3VAR_2IN_1OUT(long3, +)
DECLOP_3VAR_2IN_1OUT(long3, -)
DECLOP_3VAR_2IN_1OUT(long3, *)
DECLOP_3VAR_2IN_1OUT(long3, /)
DECLOP_3VAR_2IN_1OUT(long3, %)
DECLOP_3VAR_2IN_1OUT(long3, &)
DECLOP_3VAR_2IN_1OUT(long3, |)
DECLOP_3VAR_2IN_1OUT(long3, ^)
DECLOP_3VAR_2IN_1OUT(long3, <<)
DECLOP_3VAR_2IN_1OUT(long3, >>)

DECLOP_3VAR_ASSIGN(long3, +=)
DECLOP_3VAR_ASSIGN(long3, -=)
DECLOP_3VAR_ASSIGN(long3, *=)
DECLOP_3VAR_ASSIGN(long3, /=)
DECLOP_3VAR_ASSIGN(long3, %=)
DECLOP_3VAR_ASSIGN(long3, &=)
DECLOP_3VAR_ASSIGN(long3, |=)
DECLOP_3VAR_ASSIGN(long3, ^=)
DECLOP_3VAR_ASSIGN(long3, <<=)
DECLOP_3VAR_ASSIGN(long3, >>=)

DECLOP_3VAR_PREOP(long3, ++)
DECLOP_3VAR_PREOP(long3, --)

DECLOP_3VAR_POSTOP(long3, ++)
DECLOP_3VAR_POSTOP(long3, --)

DECLOP_3VAR_COMP(long3, ==)
DECLOP_3VAR_COMP(long3, !=)
DECLOP_3VAR_COMP(long3, <)
DECLOP_3VAR_COMP(long3, >)
DECLOP_3VAR_COMP(long3, <=)
DECLOP_3VAR_COMP(long3, >=)

DECLOP_3VAR_COMP(long3, &&)
DECLOP_3VAR_COMP(long3, ||)

DECLOP_3VAR_1IN_1OUT(long3, ~)
DECLOP_3VAR_1IN_BOOLOUT(long3, !)

DECLOP_3VAR_SCALE_PRODUCT(long3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(long3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(long3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(long3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(long3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(long3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(long3, float)
DECLOP_3VAR_SCALE_PRODUCT(long3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(long3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(long3, double)
DECLOP_3VAR_SCALE_PRODUCT(long3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(long3, signed long long)

// SIGNED LONG4

DECLOP_4VAR_2IN_1OUT(long4, +)
DECLOP_4VAR_2IN_1OUT(long4, -)
DECLOP_4VAR_2IN_1OUT(long4, *)
DECLOP_4VAR_2IN_1OUT(long4, /)
DECLOP_4VAR_2IN_1OUT(long4, %)
DECLOP_4VAR_2IN_1OUT(long4, &)
DECLOP_4VAR_2IN_1OUT(long4, |)
DECLOP_4VAR_2IN_1OUT(long4, ^)
DECLOP_4VAR_2IN_1OUT(long4, <<)
DECLOP_4VAR_2IN_1OUT(long4, >>)

DECLOP_4VAR_ASSIGN(long4, +=)
DECLOP_4VAR_ASSIGN(long4, -=)
DECLOP_4VAR_ASSIGN(long4, *=)
DECLOP_4VAR_ASSIGN(long4, /=)
DECLOP_4VAR_ASSIGN(long4, %=)
DECLOP_4VAR_ASSIGN(long4, &=)
DECLOP_4VAR_ASSIGN(long4, |=)
DECLOP_4VAR_ASSIGN(long4, ^=)
DECLOP_4VAR_ASSIGN(long4, <<=)
DECLOP_4VAR_ASSIGN(long4, >>=)

DECLOP_4VAR_PREOP(long4, ++)
DECLOP_4VAR_PREOP(long4, --)

DECLOP_4VAR_POSTOP(long4, ++)
DECLOP_4VAR_POSTOP(long4, --)

DECLOP_4VAR_COMP(long4, ==)
DECLOP_4VAR_COMP(long4, !=)
DECLOP_4VAR_COMP(long4, <)
DECLOP_4VAR_COMP(long4, >)
DECLOP_4VAR_COMP(long4, <=)
DECLOP_4VAR_COMP(long4, >=)

DECLOP_4VAR_COMP(long4, &&)
DECLOP_4VAR_COMP(long4, ||)

DECLOP_4VAR_1IN_1OUT(long4, ~)
DECLOP_4VAR_1IN_BOOLOUT(long4, !)

DECLOP_4VAR_SCALE_PRODUCT(long4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(long4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(long4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(long4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(long4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(long4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(long4, float)
DECLOP_4VAR_SCALE_PRODUCT(long4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(long4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(long4, double)
DECLOP_4VAR_SCALE_PRODUCT(long4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(long4, signed long long)

// UNSIGNED LONGLONG1

DECLOP_1VAR_2IN_1OUT(ulonglong1, +)
DECLOP_1VAR_2IN_1OUT(ulonglong1, -)
DECLOP_1VAR_2IN_1OUT(ulonglong1, *)
DECLOP_1VAR_2IN_1OUT(ulonglong1, /)
DECLOP_1VAR_2IN_1OUT(ulonglong1, %)
DECLOP_1VAR_2IN_1OUT(ulonglong1, &)
DECLOP_1VAR_2IN_1OUT(ulonglong1, |)
DECLOP_1VAR_2IN_1OUT(ulonglong1, ^)
DECLOP_1VAR_2IN_1OUT(ulonglong1, <<)
DECLOP_1VAR_2IN_1OUT(ulonglong1, >>)


DECLOP_1VAR_ASSIGN(ulonglong1, +=)
DECLOP_1VAR_ASSIGN(ulonglong1, -=)
DECLOP_1VAR_ASSIGN(ulonglong1, *=)
DECLOP_1VAR_ASSIGN(ulonglong1, /=)
DECLOP_1VAR_ASSIGN(ulonglong1, %=)
DECLOP_1VAR_ASSIGN(ulonglong1, &=)
DECLOP_1VAR_ASSIGN(ulonglong1, |=)
DECLOP_1VAR_ASSIGN(ulonglong1, ^=)
DECLOP_1VAR_ASSIGN(ulonglong1, <<=)
DECLOP_1VAR_ASSIGN(ulonglong1, >>=)

DECLOP_1VAR_PREOP(ulonglong1, ++)
DECLOP_1VAR_PREOP(ulonglong1, --)

DECLOP_1VAR_POSTOP(ulonglong1, ++)
DECLOP_1VAR_POSTOP(ulonglong1, --)

DECLOP_1VAR_COMP(ulonglong1, ==)
DECLOP_1VAR_COMP(ulonglong1, !=)
DECLOP_1VAR_COMP(ulonglong1, <)
DECLOP_1VAR_COMP(ulonglong1, >)
DECLOP_1VAR_COMP(ulonglong1, <=)
DECLOP_1VAR_COMP(ulonglong1, >=)

DECLOP_1VAR_COMP(ulonglong1, &&)
DECLOP_1VAR_COMP(ulonglong1, ||)

DECLOP_1VAR_1IN_1OUT(ulonglong1, ~)
DECLOP_1VAR_1IN_BOOLOUT(ulonglong1, !)

DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, float)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, double)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(ulonglong1, signed long long)

// UNSIGNED LONGLONG2

DECLOP_2VAR_2IN_1OUT(ulonglong2, +)
DECLOP_2VAR_2IN_1OUT(ulonglong2, -)
DECLOP_2VAR_2IN_1OUT(ulonglong2, *)
DECLOP_2VAR_2IN_1OUT(ulonglong2, /)
DECLOP_2VAR_2IN_1OUT(ulonglong2, %)
DECLOP_2VAR_2IN_1OUT(ulonglong2, &)
DECLOP_2VAR_2IN_1OUT(ulonglong2, |)
DECLOP_2VAR_2IN_1OUT(ulonglong2, ^)
DECLOP_2VAR_2IN_1OUT(ulonglong2, <<)
DECLOP_2VAR_2IN_1OUT(ulonglong2, >>)

DECLOP_2VAR_ASSIGN(ulonglong2, +=)
DECLOP_2VAR_ASSIGN(ulonglong2, -=)
DECLOP_2VAR_ASSIGN(ulonglong2, *=)
DECLOP_2VAR_ASSIGN(ulonglong2, /=)
DECLOP_2VAR_ASSIGN(ulonglong2, %=)
DECLOP_2VAR_ASSIGN(ulonglong2, &=)
DECLOP_2VAR_ASSIGN(ulonglong2, |=)
DECLOP_2VAR_ASSIGN(ulonglong2, ^=)
DECLOP_2VAR_ASSIGN(ulonglong2, <<=)
DECLOP_2VAR_ASSIGN(ulonglong2, >>=)

DECLOP_2VAR_PREOP(ulonglong2, ++)
DECLOP_2VAR_PREOP(ulonglong2, --)

DECLOP_2VAR_POSTOP(ulonglong2, ++)
DECLOP_2VAR_POSTOP(ulonglong2, --)

DECLOP_2VAR_COMP(ulonglong2, ==)
DECLOP_2VAR_COMP(ulonglong2, !=)
DECLOP_2VAR_COMP(ulonglong2, <)
DECLOP_2VAR_COMP(ulonglong2, >)
DECLOP_2VAR_COMP(ulonglong2, <=)
DECLOP_2VAR_COMP(ulonglong2, >=)

DECLOP_2VAR_COMP(ulonglong2, &&)
DECLOP_2VAR_COMP(ulonglong2, ||)

DECLOP_2VAR_1IN_1OUT(ulonglong2, ~)
DECLOP_2VAR_1IN_BOOLOUT(ulonglong2, !)

DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, float)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, double)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(ulonglong2, signed long long)

// UNSIGNED LONGLONG3

DECLOP_3VAR_2IN_1OUT(ulonglong3, +)
DECLOP_3VAR_2IN_1OUT(ulonglong3, -)
DECLOP_3VAR_2IN_1OUT(ulonglong3, *)
DECLOP_3VAR_2IN_1OUT(ulonglong3, /)
DECLOP_3VAR_2IN_1OUT(ulonglong3, %)
DECLOP_3VAR_2IN_1OUT(ulonglong3, &)
DECLOP_3VAR_2IN_1OUT(ulonglong3, |)
DECLOP_3VAR_2IN_1OUT(ulonglong3, ^)
DECLOP_3VAR_2IN_1OUT(ulonglong3, <<)
DECLOP_3VAR_2IN_1OUT(ulonglong3, >>)

DECLOP_3VAR_ASSIGN(ulonglong3, +=)
DECLOP_3VAR_ASSIGN(ulonglong3, -=)
DECLOP_3VAR_ASSIGN(ulonglong3, *=)
DECLOP_3VAR_ASSIGN(ulonglong3, /=)
DECLOP_3VAR_ASSIGN(ulonglong3, %=)
DECLOP_3VAR_ASSIGN(ulonglong3, &=)
DECLOP_3VAR_ASSIGN(ulonglong3, |=)
DECLOP_3VAR_ASSIGN(ulonglong3, ^=)
DECLOP_3VAR_ASSIGN(ulonglong3, <<=)
DECLOP_3VAR_ASSIGN(ulonglong3, >>=)

DECLOP_3VAR_PREOP(ulonglong3, ++)
DECLOP_3VAR_PREOP(ulonglong3, --)

DECLOP_3VAR_POSTOP(ulonglong3, ++)
DECLOP_3VAR_POSTOP(ulonglong3, --)

DECLOP_3VAR_COMP(ulonglong3, ==)
DECLOP_3VAR_COMP(ulonglong3, !=)
DECLOP_3VAR_COMP(ulonglong3, <)
DECLOP_3VAR_COMP(ulonglong3, >)
DECLOP_3VAR_COMP(ulonglong3, <=)
DECLOP_3VAR_COMP(ulonglong3, >=)

DECLOP_3VAR_COMP(ulonglong3, &&)
DECLOP_3VAR_COMP(ulonglong3, ||)

DECLOP_3VAR_1IN_1OUT(ulonglong3, ~)
DECLOP_3VAR_1IN_BOOLOUT(ulonglong3, !)

DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, float)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, double)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(ulonglong3, signed long long)

// UNSIGNED LONGLONG4

DECLOP_4VAR_2IN_1OUT(ulonglong4, +)
DECLOP_4VAR_2IN_1OUT(ulonglong4, -)
DECLOP_4VAR_2IN_1OUT(ulonglong4, *)
DECLOP_4VAR_2IN_1OUT(ulonglong4, /)
DECLOP_4VAR_2IN_1OUT(ulonglong4, %)
DECLOP_4VAR_2IN_1OUT(ulonglong4, &)
DECLOP_4VAR_2IN_1OUT(ulonglong4, |)
DECLOP_4VAR_2IN_1OUT(ulonglong4, ^)
DECLOP_4VAR_2IN_1OUT(ulonglong4, <<)
DECLOP_4VAR_2IN_1OUT(ulonglong4, >>)

DECLOP_4VAR_ASSIGN(ulonglong4, +=)
DECLOP_4VAR_ASSIGN(ulonglong4, -=)
DECLOP_4VAR_ASSIGN(ulonglong4, *=)
DECLOP_4VAR_ASSIGN(ulonglong4, /=)
DECLOP_4VAR_ASSIGN(ulonglong4, %=)
DECLOP_4VAR_ASSIGN(ulonglong4, &=)
DECLOP_4VAR_ASSIGN(ulonglong4, |=)
DECLOP_4VAR_ASSIGN(ulonglong4, ^=)
DECLOP_4VAR_ASSIGN(ulonglong4, <<=)
DECLOP_4VAR_ASSIGN(ulonglong4, >>=)

DECLOP_4VAR_PREOP(ulonglong4, ++)
DECLOP_4VAR_PREOP(ulonglong4, --)

DECLOP_4VAR_POSTOP(ulonglong4, ++)
DECLOP_4VAR_POSTOP(ulonglong4, --)

DECLOP_4VAR_COMP(ulonglong4, ==)
DECLOP_4VAR_COMP(ulonglong4, !=)
DECLOP_4VAR_COMP(ulonglong4, <)
DECLOP_4VAR_COMP(ulonglong4, >)
DECLOP_4VAR_COMP(ulonglong4, <=)
DECLOP_4VAR_COMP(ulonglong4, >=)

DECLOP_4VAR_COMP(ulonglong4, &&)
DECLOP_4VAR_COMP(ulonglong4, ||)

DECLOP_4VAR_1IN_1OUT(ulonglong4, ~)
DECLOP_4VAR_1IN_BOOLOUT(ulonglong4, !)

DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, float)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, double)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(ulonglong4, signed long long)

// SIGNED LONGLONG1

DECLOP_1VAR_2IN_1OUT(longlong1, +)
DECLOP_1VAR_2IN_1OUT(longlong1, -)
DECLOP_1VAR_2IN_1OUT(longlong1, *)
DECLOP_1VAR_2IN_1OUT(longlong1, /)
DECLOP_1VAR_2IN_1OUT(longlong1, %)
DECLOP_1VAR_2IN_1OUT(longlong1, &)
DECLOP_1VAR_2IN_1OUT(longlong1, |)
DECLOP_1VAR_2IN_1OUT(longlong1, ^)
DECLOP_1VAR_2IN_1OUT(longlong1, <<)
DECLOP_1VAR_2IN_1OUT(longlong1, >>)


DECLOP_1VAR_ASSIGN(longlong1, +=)
DECLOP_1VAR_ASSIGN(longlong1, -=)
DECLOP_1VAR_ASSIGN(longlong1, *=)
DECLOP_1VAR_ASSIGN(longlong1, /=)
DECLOP_1VAR_ASSIGN(longlong1, %=)
DECLOP_1VAR_ASSIGN(longlong1, &=)
DECLOP_1VAR_ASSIGN(longlong1, |=)
DECLOP_1VAR_ASSIGN(longlong1, ^=)
DECLOP_1VAR_ASSIGN(longlong1, <<=)
DECLOP_1VAR_ASSIGN(longlong1, >>=)

DECLOP_1VAR_PREOP(longlong1, ++)
DECLOP_1VAR_PREOP(longlong1, --)

DECLOP_1VAR_POSTOP(longlong1, ++)
DECLOP_1VAR_POSTOP(longlong1, --)

DECLOP_1VAR_COMP(longlong1, ==)
DECLOP_1VAR_COMP(longlong1, !=)
DECLOP_1VAR_COMP(longlong1, <)
DECLOP_1VAR_COMP(longlong1, >)
DECLOP_1VAR_COMP(longlong1, <=)
DECLOP_1VAR_COMP(longlong1, >=)

DECLOP_1VAR_COMP(longlong1, &&)
DECLOP_1VAR_COMP(longlong1, ||)

DECLOP_1VAR_1IN_1OUT(longlong1, ~)
DECLOP_1VAR_1IN_BOOLOUT(longlong1, !)

DECLOP_1VAR_SCALE_PRODUCT(longlong1, unsigned char)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, signed char)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, unsigned short)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, signed short)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, unsigned int)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, signed int)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, float)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, unsigned long)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, signed long)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, double)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, unsigned long long)
DECLOP_1VAR_SCALE_PRODUCT(longlong1, signed long long)

// SIGNED LONGLONG2

DECLOP_2VAR_2IN_1OUT(longlong2, +)
DECLOP_2VAR_2IN_1OUT(longlong2, -)
DECLOP_2VAR_2IN_1OUT(longlong2, *)
DECLOP_2VAR_2IN_1OUT(longlong2, /)
DECLOP_2VAR_2IN_1OUT(longlong2, %)
DECLOP_2VAR_2IN_1OUT(longlong2, &)
DECLOP_2VAR_2IN_1OUT(longlong2, |)
DECLOP_2VAR_2IN_1OUT(longlong2, ^)
DECLOP_2VAR_2IN_1OUT(longlong2, <<)
DECLOP_2VAR_2IN_1OUT(longlong2, >>)

DECLOP_2VAR_ASSIGN(longlong2, +=)
DECLOP_2VAR_ASSIGN(longlong2, -=)
DECLOP_2VAR_ASSIGN(longlong2, *=)
DECLOP_2VAR_ASSIGN(longlong2, /=)
DECLOP_2VAR_ASSIGN(longlong2, %=)
DECLOP_2VAR_ASSIGN(longlong2, &=)
DECLOP_2VAR_ASSIGN(longlong2, |=)
DECLOP_2VAR_ASSIGN(longlong2, ^=)
DECLOP_2VAR_ASSIGN(longlong2, <<=)
DECLOP_2VAR_ASSIGN(longlong2, >>=)

DECLOP_2VAR_PREOP(longlong2, ++)
DECLOP_2VAR_PREOP(longlong2, --)

DECLOP_2VAR_POSTOP(longlong2, ++)
DECLOP_2VAR_POSTOP(longlong2, --)

DECLOP_2VAR_COMP(longlong2, ==)
DECLOP_2VAR_COMP(longlong2, !=)
DECLOP_2VAR_COMP(longlong2, <)
DECLOP_2VAR_COMP(longlong2, >)
DECLOP_2VAR_COMP(longlong2, <=)
DECLOP_2VAR_COMP(longlong2, >=)

DECLOP_2VAR_COMP(longlong2, &&)
DECLOP_2VAR_COMP(longlong2, ||)

DECLOP_2VAR_1IN_1OUT(longlong2, ~)
DECLOP_2VAR_1IN_BOOLOUT(longlong2, !)

DECLOP_2VAR_SCALE_PRODUCT(longlong2, unsigned char)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, signed char)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, unsigned short)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, signed short)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, unsigned int)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, signed int)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, float)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, unsigned long)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, signed long)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, double)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, unsigned long long)
DECLOP_2VAR_SCALE_PRODUCT(longlong2, signed long long)

// SIGNED LONGLONG3

DECLOP_3VAR_2IN_1OUT(longlong3, +)
DECLOP_3VAR_2IN_1OUT(longlong3, -)
DECLOP_3VAR_2IN_1OUT(longlong3, *)
DECLOP_3VAR_2IN_1OUT(longlong3, /)
DECLOP_3VAR_2IN_1OUT(longlong3, %)
DECLOP_3VAR_2IN_1OUT(longlong3, &)
DECLOP_3VAR_2IN_1OUT(longlong3, |)
DECLOP_3VAR_2IN_1OUT(longlong3, ^)
DECLOP_3VAR_2IN_1OUT(longlong3, <<)
DECLOP_3VAR_2IN_1OUT(longlong3, >>)

DECLOP_3VAR_ASSIGN(longlong3, +=)
DECLOP_3VAR_ASSIGN(longlong3, -=)
DECLOP_3VAR_ASSIGN(longlong3, *=)
DECLOP_3VAR_ASSIGN(longlong3, /=)
DECLOP_3VAR_ASSIGN(longlong3, %=)
DECLOP_3VAR_ASSIGN(longlong3, &=)
DECLOP_3VAR_ASSIGN(longlong3, |=)
DECLOP_3VAR_ASSIGN(longlong3, ^=)
DECLOP_3VAR_ASSIGN(longlong3, <<=)
DECLOP_3VAR_ASSIGN(longlong3, >>=)

DECLOP_3VAR_PREOP(longlong3, ++)
DECLOP_3VAR_PREOP(longlong3, --)

DECLOP_3VAR_POSTOP(longlong3, ++)
DECLOP_3VAR_POSTOP(longlong3, --)

DECLOP_3VAR_COMP(longlong3, ==)
DECLOP_3VAR_COMP(longlong3, !=)
DECLOP_3VAR_COMP(longlong3, <)
DECLOP_3VAR_COMP(longlong3, >)
DECLOP_3VAR_COMP(longlong3, <=)
DECLOP_3VAR_COMP(longlong3, >=)

DECLOP_3VAR_COMP(longlong3, &&)
DECLOP_3VAR_COMP(longlong3, ||)

DECLOP_3VAR_1IN_1OUT(longlong3, ~)
DECLOP_3VAR_1IN_BOOLOUT(longlong3, !)

DECLOP_3VAR_SCALE_PRODUCT(longlong3, unsigned char)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, signed char)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, unsigned short)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, signed short)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, unsigned int)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, signed int)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, float)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, unsigned long)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, signed long)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, double)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, unsigned long long)
DECLOP_3VAR_SCALE_PRODUCT(longlong3, signed long long)

// SIGNED LONGLONG4

DECLOP_4VAR_2IN_1OUT(longlong4, +)
DECLOP_4VAR_2IN_1OUT(longlong4, -)
DECLOP_4VAR_2IN_1OUT(longlong4, *)
DECLOP_4VAR_2IN_1OUT(longlong4, /)
DECLOP_4VAR_2IN_1OUT(longlong4, %)
DECLOP_4VAR_2IN_1OUT(longlong4, &)
DECLOP_4VAR_2IN_1OUT(longlong4, |)
DECLOP_4VAR_2IN_1OUT(longlong4, ^)
DECLOP_4VAR_2IN_1OUT(longlong4, <<)
DECLOP_4VAR_2IN_1OUT(longlong4, >>)

DECLOP_4VAR_ASSIGN(longlong4, +=)
DECLOP_4VAR_ASSIGN(longlong4, -=)
DECLOP_4VAR_ASSIGN(longlong4, *=)
DECLOP_4VAR_ASSIGN(longlong4, /=)
DECLOP_4VAR_ASSIGN(longlong4, %=)
DECLOP_4VAR_ASSIGN(longlong4, &=)
DECLOP_4VAR_ASSIGN(longlong4, |=)
DECLOP_4VAR_ASSIGN(longlong4, ^=)
DECLOP_4VAR_ASSIGN(longlong4, <<=)
DECLOP_4VAR_ASSIGN(longlong4, >>=)

DECLOP_4VAR_PREOP(longlong4, ++)
DECLOP_4VAR_PREOP(longlong4, --)

DECLOP_4VAR_POSTOP(longlong4, ++)
DECLOP_4VAR_POSTOP(longlong4, --)

DECLOP_4VAR_COMP(longlong4, ==)
DECLOP_4VAR_COMP(longlong4, !=)
DECLOP_4VAR_COMP(longlong4, <)
DECLOP_4VAR_COMP(longlong4, >)
DECLOP_4VAR_COMP(longlong4, <=)
DECLOP_4VAR_COMP(longlong4, >=)

DECLOP_4VAR_COMP(longlong4, &&)
DECLOP_4VAR_COMP(longlong4, ||)

DECLOP_4VAR_1IN_1OUT(longlong4, ~)
DECLOP_4VAR_1IN_BOOLOUT(longlong4, !)

DECLOP_4VAR_SCALE_PRODUCT(longlong4, unsigned char)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, signed char)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, unsigned short)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, signed short)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, unsigned int)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, signed int)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, float)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, unsigned long)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, signed long)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, double)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, unsigned long long)
DECLOP_4VAR_SCALE_PRODUCT(longlong4, signed long long)


#endif


#endif
