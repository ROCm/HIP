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

#if __clang_major__ == 4

typedef __fp16 __half;

typedef struct __attribute__((aligned(4))){
  union {
    __half p[2];
    unsigned int q;
  };
} __half2;

struct hipHalfHolder{
  union {
    __half h;
    unsigned short s;
  };
};

#define HINF 65504

static struct hipHalfHolder __hInfValue = {HINF};

extern "C" __half __hip_hc_ir_hadd_half(__half, __half);
extern "C" __half __hip_hc_ir_hfma_half(__half, __half, __half);
extern "C" __half __hip_hc_ir_hmul_half(__half, __half);
extern "C" __half __hip_hc_ir_hsub_half(__half, __half);

extern "C" int __hip_hc_ir_hadd2_int(int, int);
extern "C" int __hip_hc_ir_hfma2_int(int, int, int);
extern "C" int __hip_hc_ir_hmul2_int(int, int);
extern "C" int __hip_hc_ir_hsub2_int(int, int);

__device__ static inline __half __hadd(const __half a, const __half b) {
  return __hip_hc_ir_hadd_half(a, b);
}

__device__ static inline __half __hadd_sat(__half a, __half b) {
  return __hip_hc_ir_hadd_half(a, b);
}

__device__ static inline __half __hfma(__half a, __half b, __half c) {
  return __hip_hc_ir_hfma_half(a, b, c);
}

__device__ static inline __half __hfma_sat(__half a, __half b, __half c) {
  return __hip_hc_ir_hfma_half(a, b, c);
}

__device__ static inline __half __hmul(__half a, __half b) {
  return __hip_hc_ir_hmul_half(a, b);
}

__device__ static inline __half __hmul_sat(__half a, __half b) {
  return __hip_hc_ir_hmul_half(a, b);
}

__device__ static inline __half __hneg(__half a) {
  return -a;
}

__device__ static inline __half __hsub(__half a, __half b) {
  return __hip_hc_ir_hsub_half(a, b);
}

__device__ static inline __half __hsub_sat(__half a, __half b) {
  return __hip_hc_ir_hsub_half(a, b);
}

__device__ static inline __half hdiv(__half a, __half b) {
  return a/b;
}

/*
  Half2 Arithmetic Functions
*/

__device__ static inline __half2 __hadd2(__half2 a, __half2 b) {
  __half2 c;
  c.q = __hip_hc_ir_hadd2_int(a.q, b.q);
  return c;
}

__device__ static inline __half2 __hadd2_sat(__half2 a, __half2 b) {
  __half2 c;
  c.q = __hip_hc_ir_hadd2_int(a.q, b.q);
  return c;
}

__device__ static inline __half2 __hfma2(__half2 a, __half2 b, __half2 c) {
  __half2 d;
  d.q = __hip_hc_ir_hfma2_int(a.q, b.q, c.q);
  return d;
}

__device__ static inline __half2 __hfma2_sat(__half2 a, __half2 b, __half2 c) {
  __half2 d;
  d.q = __hip_hc_ir_hfma2_int(a.q, b.q, c.q);
  return d;
}

__device__ static inline __half2 __hmul2(__half2 a, __half2 b) {
  __half2 c;
  c.q = __hip_hc_ir_hmul2_int(a.q, b.q);
  return c;
}

__device__ static inline __half2 __hmul2_sat(__half2 a, __half2 b) {
  __half2 c;
  c.q = __hip_hc_ir_hmul2_int(a.q, b.q);
  return c;
}

__device__ static inline __half2 __hsub2(__half2 a, __half2 b) {
  __half2 c;
  c.q = __hip_hc_ir_hsub2_int(a.q, b.q);
  return c;
}

__device__ static inline __half2 __hneg2(__half2 a) {
  __half2 c;
  c.p[0] = - a.p[0];
  c.p[1] = - a.p[1];
  return c;
}

__device__ static inline __half2 __hsub2_sat(__half2 a, __half2 b) {
  __half2 c;
  c.q = __hip_hc_ir_hsub2_int(a.q, b.q);
  return c;
}

__device__ static inline __half2 h2div(__half2 a, __half2 b) {
  __half2 c;
  c.p[0] = a.p[0] / b.p[0];
  c.p[1] = a.p[1] / b.p[1];
  return c;
}

/*
Half comparision Functions
*/

__device__ static inline bool __heq(__half a, __half b) {
  return a == b ? true : false;
}

__device__ static inline bool __hge(__half a, __half b) {
  return a >= b ? true : false;
}

__device__ static inline bool __hgt(__half a, __half b) {
  return a > b ? true : false;
}

__device__ static inline bool __hisinf(__half a) {
  return a == __hInfValue.h ? true : false;
}

__device__ static inline bool __hisnan(__half a) {
  return a > __hInfValue.h ? true : false;
}

__device__ static inline bool __hle(__half a, __half b) {
  return a <= b ? true : false;
}

__device__ static inline bool __hlt(__half a, __half b) {
  return a < b ? true : false;
}

__device__ static inline bool __hne(__half a, __half b) {
  return a != b ? true : false;
}

/*
Half2 Comparision Functions
*/

__device__ static inline bool __hbeq2(__half2 a, __half2 b) {
  return (a.p[0] == b.p[0] ? true : false) && (a.p[1] == b.p[1] ? true : false);
}

__device__ static inline bool __hbge2(__half2 a, __half2 b) {
  return (a.p[0] >= b.p[0] ? true : false) && (a.p[1] >= b.p[1] ? true : false);
}

__device__ static inline bool __hbgt2(__half2 a, __half2 b) {
  return (a.p[0] > b.p[0] ? true : false) && (a.p[1] > b.p[1] ? true : false);
}

__device__ static inline bool __hble2(__half2 a, __half2 b) {
  return (a.p[0] <= b.p[0] ? true : false) && (a.p[1] <= b.p[1] ? true : false);
}

__device__ static inline bool __hblt2(__half2 a, __half2 b) {
  return (a.p[0] < b.p[0] ? true : false) && (a.p[1] < b.p[1] ? true : false);
}

__device__ static inline bool __hbne2(__half2 a, __half2 b) {
  return (a.p[0] != b.p[0] ? true : false) && (a.p[1] != b.p[1] ? true : false);
}

__device__ static inline __half2 __heq2(__half2 a, __half2 b) {
  __half2 c;
  c.p[0] = (a.p[0] == b.p[0]) ? (__half)1 : (__half)0;
  c.p[1] = (a.p[1] == b.p[1]) ? (__half)1 : (__half)0;
  return c;
}

__device__ static inline __half2 __hge2(__half2 a, __half2 b) {
  __half2 c;
  c.p[0] = (a.p[0] >= b.p[0]) ? (__half)1 : (__half)0;
  c.p[1] = (a.p[1] >= b.p[1]) ? (__half)1 : (__half)0;
  return c;
}

__device__ static inline __half2 __hgt2(__half2 a, __half2 b) {
  __half2 c;
  c.p[0] = (a.p[0] > b.p[0]) ? (__half)1 : (__half)0;
  c.p[1] = (a.p[1] > b.p[1]) ? (__half)1 : (__half)0;
  return c;
}

__device__ static inline __half2 __hisnan2(__half2 a) {
  __half2 c;
  c.p[0] = (a.p[0] > __hInfValue.h) ? (__half)1 : (__half)0;
  c.p[1] = (a.p[1] > __hInfValue.h) ? (__half)1 : (__half)0;
  return c;
}

__device__ static inline __half2 __hle2(__half2 a, __half2 b) {
  __half2 c;
  c.p[0] = (a.p[0] <= b.p[0]) ? (__half)1 : (__half)0;
  c.p[1] = (a.p[1] <= b.p[1]) ? (__half)1 : (__half)0;
  return c;
}

__device__ static inline __half2 __hlt2(__half2 a, __half2 b) {
  __half2 c;
  c.p[0] = (a.p[0] < b.p[0]) ? (__half)1 : (__half)0;
  c.p[1] = (a.p[1] < b.p[1]) ? (__half)1 : (__half)0;
  return c;
}

__device__ static inline __half2 __hne2(__half2 a, __half2 b) {
  __half2 c;
  c.p[0] = (a.p[0] != b.p[0]) ? (__half)1 : (__half)0;
  c.p[1] = (a.p[1] != b.p[1]) ? (__half)1 : (__half)0;
  return c;
}

/*
Conversion instructions
*/

__device__ static inline __half2 __float22half2_rn(const float2 a) {
  __half2 b;
  b.p[0] = (__half)a.x;
  b.p[1] = (__half)a.y;
  return b;
}

__device__ static inline __half __float2half(const float a) {
  return (__half)a;
}

__device__ static inline __half2 __float2half2_rn(const float a) {
  __half2 b;
  b.p[0] = (__half)a;
  b.p[1] = (__half)a;
  return b;
}

__device__ static inline __half __float2half_rd(const float a) {
  return (__half)a;
}

__device__ static inline __half __float2half_ru(const float a) {
  return (__half)a;
}

__device__ static inline __half __float2half_rz(const float a) {
  return (__half)a;
}

__device__ static inline __half2 __floats2half2_rn(const float a, const float b) {
  __half2 c;
  c.p[0] = (__half)a;
  c.p[1] = (__half)b;
  return c;
}

__device__ static inline float2 __half22float2(const __half2 a) {
  float2 b;
  b.x = (float)a.p[0];
  b.y = (float)a.p[1];
  return b;
}

__device__ static inline float __half2float(const __half a) {
  return (float)a;
}

__device__ static inline __half2 half2half2(const __half a) {
  __half2 b;
  b.p[0] = a;
  b.p[1] = a;
  return b;
}

__device__ static inline int __half2int_rd(__half h) {
  return (int)h;
}

__device__ static inline int __half2int_rn(__half h) {
  return (int)h;
}

__device__ static inline int __half2int_ru(__half h) {
  return (int)h;
}

__device__ static inline int __half2int_rz(__half h) {
  return (int)h;
}

__device__ static inline long long int __half2ll_rd(__half h) {
  return (long long int)h;
}

__device__ static inline long long int __half2ll_rn(__half h) {
  return (long long int)h;
}

__device__ static inline long long int __half2ll_ru(__half h) {
  return (long long int)h;
}

__device__ static inline long long int __half2ll_rz(__half h) {
  return (long long int)h;
}

__device__ static inline short __half2short_rd(__half h) {
  return (short)h;
}

__device__ static inline short __half2short_rn(__half h) {
  return (short)h;
}

__device__ static inline short __half2short_ru(__half h) {
  return (short)h;
}

__device__ static inline short __half2short_rz(__half h) {
  return (short)h;
}

__device__ static inline unsigned int __half2uint_rd(__half h) {
  return (unsigned int)h;
}

__device__ static inline unsigned int __half2uint_rn(__half h) {
  return (unsigned int)h;
}

__device__ static inline unsigned int __half2uint_ru(__half h) {
  return (unsigned int)h;
}

__device__ static inline unsigned int __half2uint_rz(__half h) {
  return (unsigned int)h;
}

__device__ static inline unsigned long long int __half2ull_rd(__half h) {
  return (unsigned long long)h;
}

__device__ static inline unsigned long long int __half2ull_rn(__half h) {
  return (unsigned long long)h;
}

__device__ static inline unsigned long long int __half2ull_ru(__half h) {
  return (unsigned long long)h;
}

__device__ static inline unsigned long long int __half2ull_rz(__half h) {
  return (unsigned long long)h;
}

__device__ static inline unsigned short int __half2ushort_rd(__half h) {
  return (unsigned short int)h;
}

__device__ static inline unsigned short int __half2ushort_rn(__half h) {
  return (unsigned short int)h;
}

__device__ static inline unsigned short int __half2ushort_ru(__half h) {
  return (unsigned short int)h;
}

__device__ static inline unsigned short int __half2ushort_rz(__half h) {
  return (unsigned short int)h;
}

__device__ static inline short int __half_as_short(const __half h) {
  hipHalfHolder hH;
  hH.h = h;
  return (short)hH.s;
}

__device__ static inline unsigned short int __half_as_ushort(const __half h) {
  hipHalfHolder hH;
  hH.h = h;
  return hH.s;
}

__device__ static inline __half2 __halves2half2(const __half a, const __half b) {
  __half2 c;
  c.p[0] = a;
  c.p[1] = b;
  return c;
}

__device__ static inline float __high2float(const __half2 a) {
  return (float)a.p[1];
}

__device__ static inline __half __high2half(const __half2 a) {
  return a.p[1];
}

__device__ static inline __half2 __high2half2(const __half2 a) {
  __half2 b;
  b.p[0] = a.p[1];
  b.p[1] = a.p[1];
  return b;
}

__device__ static inline __half2 __highs2half2(const __half2 a, const __half2 b) {
  __half2 c;
  c.p[0] = a.p[1];
  c.p[1] = b.p[1];
  return c;
}

__device__ static inline __half __int2half_rd(int i) {
  return (__half)i;
}

__device__ static inline __half __int2half_rn(int i) {
  return (__half)i;
}

__device__ static inline __half __int2half_ru(int i) {
  return (__half)i;
}

__device__ static inline __half __int2half_rz(int i) {
  return (__half)i;
}

__device__ static inline __half __ll2half_rd(long long int i){
  return (__half)i;
}

__device__ static inline __half __ll2half_rn(long long int i){
  return (__half)i;
}

__device__ static inline __half __ll2half_ru(long long int i){
  return (__half)i;
}

__device__ static inline __half __ll2half_rz(long long int i){
  return (__half)i;
}

__device__ static inline float __low2float(const __half2 a) {
  return (float)a.p[0];
}

__device__ static inline __half __low2half(const __half2 a) {
  return a.p[0];
}

__device__ static inline __half2 __low2half2(const __half2 a, const __half2 b) {
  __half2 c;
  c.p[0] = a.p[0];
  c.p[1] = b.p[0];
  return c;
}

__device__ static inline __half2 __low2half2(const __half2 a) {
  __half2 b;
  b.p[0] = a.p[0];
  b.p[1] = a.p[0];
  return b;
}

__device__ static inline __half2 __lowhigh2highlow(const __half2 a) {
  __half2 b;
  b.p[0] = a.p[1];
  b.p[1] = a.p[0];
  return b;
}

__device__ static inline __half2 __lows2half2(const __half2 a, const __half2 b) {
  __half2 c;
  c.p[0] = a.p[0];
  c.p[1] = b.p[0];
  return c;
}

__device__ static inline __half __short2half_rd(short int i) {
  return (__half)i;
}

__device__ static inline __half __short2half_rn(short int i) {
  return (__half)i;
}

__device__ static inline __half __short2half_ru(short int i) {
  return (__half)i;
}

__device__ static inline __half __short2half_rz(short int i) {
  return (__half)i;
}

__device__ static inline __half __uint2half_rd(unsigned int i) {
  return (__half)i;
}

__device__ static inline __half __uint2half_rn(unsigned int i) {
  return (__half)i;
}

__device__ static inline __half __uint2half_ru(unsigned int i) {
  return (__half)i;
}

__device__ static inline __half __uint2half_rz(unsigned int i) {
  return (__half)i;
}

__device__ static inline __half __ull2half_rd(unsigned long long int i) {
  return (__half)i;
}

__device__ static inline __half __ull2half_rn(unsigned long long int i) {
  return (__half)i;
}

__device__ static inline __half __ull2half_ru(unsigned long long int i) {
  return (__half)i;
}

__device__ static inline __half __ull2half_rz(unsigned long long int i) {
  return (__half)i;
}

__device__ static inline __half __ushort2half_rd(unsigned short int i) {
  return (__half)i;
}

__device__ static inline __half __ushort2half_rn(unsigned short int i) {
  return (__half)i;
}

__device__ static inline __half __ushort2half_ru(unsigned short int i) {
  return (__half)i;
}

__device__ static inline __half __ushort2half_rz(unsigned short int i) {
  return (__half)i;
}

__device__ static inline __half __ushort_as_half(const unsigned short int i) {
  hipHalfHolder hH;
  hH.s = i;
  return hH.h;
}

#endif

#if __clang_major__ == 3

typedef struct {
  unsigned x: 16;
} __half;

typedef struct __attribute__((aligned(4))){
  union {
    __half p[2];
    unsigned int q;
  };
} __half2;




#endif


#endif
