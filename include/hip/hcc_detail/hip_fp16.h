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

#ifndef HIP_FP16_H
#define HIP_FP16_H

#include "hip/hip_runtime.h"

#define __CLANG_VERSION__ __clang_major__ * 10 + __clang_minor__

#ifdef HIP_HALF_HW_SUPPORT

typedef __fp16 __half;

typedef struct __attribute__((aligned(4))){
  int a;
} __half2;

extern "C" __half __hip_hadd_gfx803(__half a, __half b);
extern "C" __half __hip_hfma_gfx803(__half a, __half b);
extern "C" __half __hip_hmul_gfx803(__half a, __half b);
extern "C" __half __hip_hsub_gfx803(__half a, __half b);

extern "C" int __hip_hadd2_gfx803(int a, int b);
extern "C" int __hip_hfma2_gfx803(int a, int b);
extern "C" int __hip_hmul2_gfx803(int a, int b);
extern "C" int __hip_hsub2_gfx803(int a, int b);

__device__ inline __half __hadd(__half a, __half b) {
  return __hip_hadd_gfx803(a, b);
}

__device__ inline __half __hadd_sat(__half a, __half b) {
  return __hip_hadd_gfx803(a, b);
}

__device__ inline __half __hfma(__half a, __half b) {
  return __hip_hfma_gfx803(a, b);
}

__device__ inline __half __hfma_sat(__half a, __half b) {
  return __hip_hfma_gfx803(a, b);
}

__device__ inline __half __hmul(__half a, __half b) {
  return __hip_hmul_gfx803(a, b);
}

__device__ inline __half __hmul_sat(__half a, __half b) {
  return __hip_hmul_gfx803(a, b);
}

__device__ inline __half __hsub(__half a, __half b) {
  return __hip_hsub_gfx803(a, b);
}

__device__ inline __half __hsub_sat(__half a, __half b) {
  return __hip_hsub_gfx803(a, b);
}


__device__ inline __half2 __hadd2(__half2 a, __half2 b) {
  __half2 ret;
  ret.a = __hip_hadd2_gfx803(a.a, b.a);
  return ret;
}

#else

typedef struct{
  unsigned x: 16;
} __half;


typedef struct __attribute__((aligned(4))){
  __half p,q;
} __half2;

typedef __half half;
typedef __half2 half2;

/*
Arithmetic functions
*/

__device__ __half __hadd(const __half a, const __half b);

__device__ __half __hadd_sat(const __half a, const __half b);

__device__ __half __hfma(const __half a, const __half b, const __half c);

__device__ __half __hfma_sat(const __half a, const __half b, const __half c);

__device__ __half __hmul(const __half a, const __half b);

__device__ __half __hmul_sat(const __half a, const __half b);

__device__ __half __hneq(const __half a);

__device__ __half __hsub(const __half a, const __half b);

__device__ __half __hsub_sat(const __half a, const __half b);



/*
Half2 Arithmetic Instructions
*/

__device__ __half2 __hadd2(const __half2 a, const __half2 b);

__device__ __half2 __hadd2_sat(const __half2 a, const __half2 b);

__device__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);

__device__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);

__device__ __half2 __hmul2(const __half2 a, const __half2 b);

__device__ __half2 __hmul2_sat(const __half2 a, const __half2 b);

__device__ __half2 __hneq2(const __half2 a);

__device__ __half2 __hsub2(const __half2 a, const __half2 b);

__device__ __half2 __hsub2_sat(const __half2 a, const __half2 b);

/*
Half Cmps
*/

__device__  bool __heq(const __half a, const __half b);

__device__ bool __hge(const __half a, const __half b);

__device__ bool __hgt(const __half a, const __half b);

__device__ bool __hisinf(const __half a);

__device__ bool __hisnan(const __half a);

__device__ bool __hle(const __half a, const __half b);

__device__ bool __hlt(const __half a, const __half b);

__device__ bool __hne(const __half a, const __half b);

/*
Half2 Cmps
*/

__device__ bool __hbeq2(const __half2 a, const __half2 b);

__device__ bool __hbge2(const __half2 a, const __half2 b);

__device__ bool __hbgt2(const __half2 a, const __half2 b);

__device__ bool __hble2(const __half2 a, const __half2 b);

__device__ bool __hblt2(const __half2 a, const __half2 b);

__device__ bool __hbne2(const __half2 a, const __half2 b);

__device__ __half2 __heq2(const __half2 a, const __half2 b);

__device__ __half2 __hge2(const __half2 a, const __half2 b);

__device__ __half2 __hgt2(const __half2 a, const __half2 b);

__device__ __half2 __hisnan2(const __half2 a);

__device__ __half2 __hle2(const __half2 a, const __half2 b);

__device__ __half2 __hlt2(const __half2 a, const __half2 b);

__device__ __half2 __hne2(const __half2 a, const __half2 b);


/*
Half Cnvs and Data Mvmnt
*/

__device__ __half2 __float22half2_rn(const float2 a);

__device__ __half __float2half(const float a);

__device__ __half2 __float2half2_rn(const float a);

__device__ __half2 __floats2half2_rn(const float a, const float b);

__device__ float2 __half22float2(const __half2 a);

__device__ float __half2float(const __half a);

__device__ __half2 __half2half2(const __half a);

__device__ __half2 __halves2half2(const __half a, const __half b);

__device__ float __high2float(const __half2 a);

__device__ __half __high2half(const __half2 a);

__device__ __half2 __high2half2(const __half2 a);

__device__ __half2 __highs2half2(const __half2 a, const __half2 b);

__device__ float __low2float(const __half2 a);

__device__ __half __low2half(const __half2 a);

__device__ __half2 __low2half2(const __half2 a);

__device__ __half2 __lows2half2(const __half2 a, const __half2 b);

__device__ __half2 __lowhigh2highlow(const __half2 a);

__device__ __half2 __low2half2(const __half2 a, const __half2 b);

#endif
#endif
