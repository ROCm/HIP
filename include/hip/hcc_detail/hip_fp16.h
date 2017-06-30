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

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_FP16_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_FP16_H

#include "hip/hcc_detail/hip_vector_types.h"
#if ( __clang_major__ > 3)
typedef __fp16 __half;
typedef __fp16 __half1 __attribute__((ext_vector_type(1)));
typedef __fp16 __half2 __attribute__((ext_vector_type(2)));
typedef __fp16 half;

/*
Half Arithmetic Functions
*/
__device__ __half __hadd(const __half a, const __half b);
__device__ __half __hadd_sat(__half a, __half b);
__device__ __half __hfma(__half a, __half b, __half c);
__device__ __half __hfma_sat(__half a, __half b, __half c);
__device__ __half __hmul(__half a, __half b);
__device__ __half __hmul_sat(__half a, __half b);
__device__ __half __hneg(__half a);
__device__ __half __hsub(__half a, __half b);
__device__ __half __hsub_sat(__half a, __half b);
__device__ __half hdiv(__half a, __half b);

/*
Half2 Arithmetic Functions
*/

__device__ static __half2 __hadd2(__half2 a, __half2 b);
__device__ static __half2 __hadd2_sat(__half2 a, __half2 b);
__device__ static __half2 __hfma2(__half2 a, __half2 b, __half2 c);
__device__ static __half2 __hfma2_sat(__half2 a, __half2 b, __half2 c);
__device__ static __half2 __hmul2(__half2 a, __half2 b);
__device__ static __half2 __hmul2_sat(__half2 a, __half2 b);
__device__ static __half2 __hsub2(__half2 a, __half2 b);
__device__ static __half2 __hneg2(__half2 a);
__device__ static __half2 __hsub2_sat(__half2 a, __half2 b);
__device__ static __half2 h2div(__half2 a, __half2 b);

/*
Half Comparision Functions
*/

__device__  bool __heq(__half a, __half b);
__device__  bool __hge(__half a, __half b);
__device__  bool __hgt(__half a, __half b);
__device__  bool __hisinf(__half a);
__device__  bool __hisnan(__half a);
__device__  bool __hle(__half a, __half b);
__device__  bool __hlt(__half a, __half b);
__device__  bool __hne(__half a, __half b);

/*
Half2 Comparision Functions
*/

__device__  bool __hbeq2(__half2 a, __half2 b);
__device__  bool __hbge2(__half2 a, __half2 b);
__device__  bool __hbgt2(__half2 a, __half2 b);
__device__  bool __hble2(__half2 a, __half2 b);
__device__  bool __hblt2(__half2 a, __half2 b);
__device__  bool __hbne2(__half2 a, __half2 b);
__device__  __half2 __heq2(__half2 a, __half2 b);
__device__  __half2 __hge2(__half2 a, __half2 b);
__device__  __half2 __hgt2(__half2 a, __half2 b);
__device__  __half2 __hisnan2(__half2 a);
__device__  __half2 __hle2(__half2 a, __half2 b);
__device__  __half2 __hlt2(__half2 a, __half2 b);
__device__  __half2 __hne2(__half2 a, __half2 b);

/*
Half Math Functions
*/

__device__ static __half hceil(const __half h);
__device__ static __half hcos(const __half h);
__device__ static __half hexp(const __half h);
__device__ static __half hexp10(const __half h);
__device__ static __half hexp2(const __half h);
__device__ static __half hfloor(const __half h);
__device__ static __half hlog(const __half h);
__device__ static __half hlog10(const __half h);
__device__ static __half hlog2(const __half h);
//__device__ static __half hrcp(const __half h);
__device__ static __half hrint(const __half h);
__device__ static __half hsin(const __half h);
__device__ static __half hsqrt(const __half a);
__device__ static __half htrunc(const __half a);

/*
Half2 Math Functions
*/

__device__ static __half2 h2ceil(const __half2 h);
__device__ static __half2 h2exp(const __half2 h);
__device__ static __half2 h2exp10(const __half2 h);
__device__ static __half2 h2exp2(const __half2 h);
__device__ static __half2 h2floor(const __half2 h);
__device__ static __half2 h2log(const __half2 h);
__device__ static __half2 h2log10(const __half2 h);
__device__ static __half2 h2log2(const __half2 h);
__device__ static __half2 h2rcp(const __half2 h);
__device__ static __half2 h2rsqrt(const __half2 h);
__device__ static __half2 h2sin(const __half2 h);
__device__ static __half2 h2sqrt(const __half2 h);

/*
Half Conversion And Data Movement
*/

__device__  __half2 __float22half2_rn(const float2 a);
__device__  __half __float2half(const float a);
__device__  __half2 __float2half2_rn(const float a);
__device__  __half __float2half_rd(const float a);
__device__  __half __float2half_rn(const float a);
__device__  __half __float2half_ru(const float a);
__device__  __half __float2half_rz(const float a);
__device__  __half2 __floats2half2_rn(const float a, const float b);
__device__  float2 __half22float2(const __half2 a);
__device__  float __half2float(const __half a);
__device__  __half2 half2half2(const __half a);
__device__  int __half2int_rd(__half h);
__device__  int __half2int_rn(__half h);
__device__  int __half2int_ru(__half h);
__device__  int __half2int_rz(__half h);
__device__  long long int __half2ll_rd(__half h);
__device__  long long int __half2ll_rn(__half h);
__device__  long long int __half2ll_ru(__half h);
__device__  long long int __half2ll_rz(__half h);
__device__  short __half2short_rd(__half h);
__device__  short __half2short_rn(__half h);
__device__  short __half2short_ru(__half h);
__device__  short __half2short_rz(__half h);
__device__  unsigned int __half2uint_rd(__half h);
__device__  unsigned int __half2uint_rn(__half h);
__device__  unsigned int __half2uint_ru(__half h);
__device__  unsigned int __half2uint_rz(__half h);
__device__  unsigned long long int __half2ull_rd(__half h);
__device__  unsigned long long int __half2ull_rn(__half h);
__device__  unsigned long long int __half2ull_ru(__half h);
__device__  unsigned long long int __half2ull_rz(__half h);
__device__  unsigned short int __half2ushort_rd(__half h);
__device__  unsigned short int __half2ushort_rn(__half h);
__device__  unsigned short int __half2ushort_ru(__half h);
__device__  unsigned short int __half2ushort_rz(__half h);
__device__  short int __half_as_short(const __half h);
__device__  unsigned short int __half_as_ushort(const __half h);
__device__  __half2 __halves2half2(const __half a, const __half b);
__device__  float __high2float(const __half2 a);
__device__  __half __high2half(const __half2 a);
__device__  __half2 __high2half2(const __half2 a);
__device__  __half2 __highs2half2(const __half2 a, const __half2 b);
__device__  __half __int2half_rd(int i);
__device__  __half __int2half_rn(int i);
__device__  __half __int2half_ru(int i);
__device__  __half __int2half_rz(int i);
__device__  __half __ll2half_rd(long long int i);
__device__  __half __ll2half_rn(long long int i);
__device__  __half __ll2half_ru(long long int i);
__device__  __half __ll2half_rz(long long int i);
__device__  float __low2float(const __half2 a);

__device__ __half __low2half(const __half2 a);
__device__ __half2 __low2half2(const __half2 a, const __half2 b);
__device__ __half2 __low2half2(const __half2 a);
__device__ __half2 __lowhigh2highlow(const __half2 a);
__device__ __half2 __lows2half2(const __half2 a, const __half2 b);
__device__  __half __short2half_rd(short int i);
__device__  __half __short2half_rn(short int i);
__device__  __half __short2half_ru(short int i);
__device__  __half __short2half_rz(short int i);
__device__  __half __uint2half_rd(unsigned int i);
__device__  __half __uint2half_rn(unsigned int i);
__device__  __half __uint2half_ru(unsigned int i);
__device__  __half __uint2half_rz(unsigned int i);
__device__  __half __ull2half_rd(unsigned long long int i);
__device__  __half __ull2half_rn(unsigned long long int i);
__device__  __half __ull2half_ru(unsigned long long int i);
__device__  __half __ull2half_rz(unsigned long long int i);
__device__  __half __ushort2half_rd(unsigned short int i);
__device__  __half __ushort2half_rn(unsigned short int i);
__device__  __half __ushort2half_ru(unsigned short int i);
__device__  __half __ushort2half_rz(unsigned short int i);
__device__  __half __ushort_as_half(const unsigned short int i);

extern "C" __half2 __hip_hc_ir_hadd2_int(__half2, __half2);
extern "C" __half2 __hip_hc_ir_hfma2_int(__half2, __half2, __half2);
extern "C" __half2 __hip_hc_ir_hmul2_int(__half2, __half2);
extern "C" __half2 __hip_hc_ir_hsub2_int(__half2, __half2);

extern "C" __half __hip_hc_ir_hceil_half(__half) __asm("llvm.ceil.f16");
extern "C" __half __hip_hc_ir_hcos_half(__half) __asm("llvm.cos.f16");
extern "C" __half __hip_hc_ir_hexp2_half(__half) __asm("llvm.exp2.f16");
extern "C" __half __hip_hc_ir_hfloor_half(__half) __asm("llvm.floor.f16");
extern "C" __half __hip_hc_ir_hlog2_half(__half) __asm("llvm.log2.f16");
extern "C" __half __hip_hc_ir_hrcp_half(__half) __asm("llvm.amdgcn.rcp.f16");
extern "C" __half __hip_hc_ir_hrint_half(__half) __asm("llvm.rint.f16");
extern "C" __half __hip_hc_ir_hrsqrt_half(__half) __asm("llvm.sqrt.f16");
extern "C" __half __hip_hc_ir_hsin_half(__half) __asm("llvm.sin.f16");
extern "C" __half __hip_hc_ir_hsqrt_half(__half) __asm("llvm.sqrt.f16");
extern "C" __half __hip_hc_ir_htrunc_half(__half) __asm("llvm.trunc.f16");

extern "C" __half2 __hip_hc_ir_h2ceil_int(__half2);
extern "C" __half2 __hip_hc_ir_h2cos_int(__half2);
extern "C" __half2 __hip_hc_ir_h2exp2_int(__half2);
extern "C" __half2 __hip_hc_ir_h2floor_int(__half2);
extern "C" __half2 __hip_hc_ir_h2log2_int(__half2);
extern "C" __half2 __hip_hc_ir_h2rcp_int(__half2);
extern "C" __half2 __hip_hc_ir_h2rsqrt_int(__half2);
extern "C" __half2 __hip_hc_ir_h2sin_int(__half2);
extern "C" __half2 __hip_hc_ir_h2sqrt_int(__half2);
extern "C" __half2 __hip_hc_ir_h2trunc_int(__half2);

/*
  Half2 Arithmetic Functions
*/

__device__ static inline __half2 __hadd2(__half2 a, __half2 b) {
  __half2 c;
  c.xy = __hip_hc_ir_hadd2_int(a.xy, b.xy);
  return c;
}

__device__ static inline __half2 __hadd2_sat(__half2 a, __half2 b) {
  __half2 c;
  c.xy = __hip_hc_ir_hadd2_int(a.xy, b.xy);
  return c;
}

__device__ static inline __half2 __hfma2(__half2 a, __half2 b, __half2 c) {
  __half2 d;
  d.xy = __hip_hc_ir_hfma2_int(a.xy, b.xy, c.xy);
  return d;
}

__device__ static inline __half2 __hfma2_sat(__half2 a, __half2 b, __half2 c) {
  __half2 d;
  d.xy = __hip_hc_ir_hfma2_int(a.xy, b.xy, c.xy);
  return d;
}

__device__ static inline __half2 __hmul2(__half2 a, __half2 b) {
  __half2 c;
  c.xy = __hip_hc_ir_hmul2_int(a.xy, b.xy);
  return c;
}

__device__ static inline __half2 __hmul2_sat(__half2 a, __half2 b) {
  __half2 c;
  c.xy = __hip_hc_ir_hmul2_int(a.xy, b.xy);
  return c;
}

__device__ static inline __half2 __hsub2(__half2 a, __half2 b) {
  __half2 c;
  c.xy = __hip_hc_ir_hsub2_int(a.xy, b.xy);
  return c;
}

__device__ static inline __half2 __hneg2(__half2 a) {
  __half2 c;
  c.x = - a.x;
  c.y = - a.y;
  return c;
}

__device__ static inline __half2 __hsub2_sat(__half2 a, __half2 b) {
  __half2 c;
  c.xy = __hip_hc_ir_hsub2_int(a.xy, b.xy);
  return c;
}

__device__ static inline __half2 h2div(__half2 a, __half2 b) {
  __half2 c;
  c.x = a.x / b.x;
  c.y = a.y / b.y;
  return c;
}


__device__ static inline __half hceil(const __half h) {
  return __hip_hc_ir_hceil_half(h);
}

__device__ static inline __half hcos(const __half h) {
  return __hip_hc_ir_hcos_half(h);
}

__device__ static inline __half hexp(const __half h) {
  return __hip_hc_ir_hexp2_half(__hmul(h, 1.442694));
}

__device__ static inline __half hexp10(const __half h) {
  return __hip_hc_ir_hexp2_half(__hmul(h, 3.3219281));
}

__device__ static inline __half hexp2(const __half h) {
  return __hip_hc_ir_hexp2_half(h);
}

__device__ static inline __half hfloor(const __half h) {
  return __hip_hc_ir_hfloor_half(h);
}

__device__ static inline __half hlog(const __half h) {
  return __hmul(__hip_hc_ir_hlog2_half(h), 0.693147);
}

__device__ static inline __half hlog10(const __half h) {
  return __hmul(__hip_hc_ir_hlog2_half(h),  0.301029);
}

__device__ static inline __half hlog2(const __half h) {
  return __hip_hc_ir_hlog2_half(h);
}
/*
__device__ static inline __half hrcp(const __half h) {
  return __hip_hc_ir_hrcp_half(h);
}
*/
__device__ static inline __half hrint(const __half h) {
  return __hip_hc_ir_hrint_half(h);
}

__device__ static inline __half hrsqrt(const __half h) {
  return __hip_hc_ir_hrsqrt_half(h);
}

__device__ static inline __half hsin(const __half h) {
  return __hip_hc_ir_hsin_half(h);
}

__device__ static inline __half hsqrt(const __half a) {
  return __hip_hc_ir_hsqrt_half(a);
}

__device__ static inline __half htrunc(const __half a) {
  return __hip_hc_ir_htrunc_half(a);
}

/*
Half2 Math Operations
*/

__device__ static inline __half2 h2ceil(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2ceil_int(h.xy);
  return a;
}

__device__ static inline __half2 h2cos(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2cos_int(h.xy);
  return a;
}

__device__ static inline __half2 h2exp(const __half2 h) {
  __half2 factor;
  factor.x = 1.442694;
  factor.y = 1.442694;
  factor.xy = __hip_hc_ir_h2exp2_int(__hip_hc_ir_hmul2_int(h.xy, factor.xy));
  return factor;
}

__device__ static inline __half2 h2exp10(const __half2 h) {
  __half2 factor;
  factor.x = 3.3219281;
  factor.y = 3.3219281;
  factor.xy = __hip_hc_ir_h2exp2_int(__hip_hc_ir_hmul2_int(h.xy, factor.xy));
  return factor;
}

__device__ static inline __half2 h2exp2(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2exp2_int(h.xy);
  return a;
}

__device__ static inline __half2 h2floor(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2floor_int(h.xy);
  return a;
}

__device__ static inline __half2 h2log(const __half2 h) {
  __half2 factor;
  factor.x = 0.693147;
  factor.y = 0.693147;
  factor.xy = __hip_hc_ir_hmul2_int(__hip_hc_ir_h2log2_int(h.xy), factor.xy);
  return factor;
}

__device__ static inline __half2 h2log10(const __half2 h) {
  __half2 factor;
  factor.x = 0.301029;
  factor.y = 0.301029;
  factor.xy = __hip_hc_ir_hmul2_int(__hip_hc_ir_h2log2_int(h.xy),  factor.xy);
  return factor;
}
__device__ static inline __half2 h2log2(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2log2_int(h.xy);
  return a;
}

__device__ static inline __half2 h2rcp(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2rcp_int(h.xy);
  return a;
}

__device__ static inline __half2 h2rsqrt(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2rsqrt_int(h.xy);
  return a;
}

__device__ static inline __half2 h2sin(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2sin_int(h.xy);
  return a;
}

__device__ static inline __half2 h2sqrt(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2sqrt_int(h.xy);
  return a;
}

__device__ static inline __half2 h2trunc(const __half2 h) {
  __half2 a;
  a.xy = __hip_hc_ir_h2trunc_int(h.xy);
  return a;
}
#endif //clang_major > 3

#endif
