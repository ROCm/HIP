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

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_DEVICE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_DEVICE_FUNCTIONS_H

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>





// Single Precision Fast Math
__device__  float __cosf(float x);
__device__  float __exp10f(float x);
__device__  float __expf(float x);
__device__ static  float __fadd_rd(float x, float y);
__device__ static  float __fadd_rn(float x, float y);
__device__ static  float __fadd_ru(float x, float y);
__device__ static  float __fadd_rz(float x, float y);
__device__ static  float __fdiv_rd(float x, float y);
__device__ static  float __fdiv_rn(float x, float y);
__device__ static  float __fdiv_ru(float x, float y);
__device__ static  float __fdiv_rz(float x, float y);
__device__ static  float __fdividef(float x, float y);
__device__  float __fmaf_rd(float x, float y, float z);
__device__  float __fmaf_rn(float x, float y, float z);
__device__  float __fmaf_ru(float x, float y, float z);
__device__  float __fmaf_rz(float x, float y, float z);
__device__ static  float __fmul_rd(float x, float y);
__device__ static  float __fmul_rn(float x, float y);
__device__ static  float __fmul_ru(float x, float y);
__device__ static  float __fmul_rz(float x, float y);
__device__  float __frcp_rd(float x);
__device__  float __frcp_rn(float x);
__device__  float __frcp_ru(float x);
__device__  float __frcp_rz(float x);
__device__  float __frsqrt_rn(float x);
__device__  float __fsqrt_rd(float x);
__device__  float __fsqrt_rn(float x);
__device__  float __fsqrt_ru(float x);
__device__  float __fsqrt_rz(float x);
__device__ static  float __fsub_rd(float x, float y);
__device__ static  float __fsub_rn(float x, float y);
__device__ static  float __fsub_ru(float x, float y);
__device__  float __log10f(float x);
__device__  float __log2f(float x);
__device__  float __logf(float x);
__device__  float __powf(float base, float exponent);
__device__ static  float __saturatef(float x);
__device__  void __sincosf(float x, float *s, float *c);
__device__  float __sinf(float x);
__device__  float __tanf(float x);


/*
Double Precision Intrinsics
*/

__device__ static  double __dadd_rd(double x, double y);
__device__ static  double __dadd_rn(double x, double y);
__device__ static  double __dadd_ru(double x, double y);
__device__ static  double __dadd_rz(double x, double y);
__device__ static  double __ddiv_rd(double x, double y);
__device__ static  double __ddiv_rn(double x, double y);
__device__ static  double __ddiv_ru(double x, double y);
__device__ static  double __ddiv_rz(double x, double y);
__device__ static  double __dmul_rd(double x, double y);
__device__ static  double __dmul_rn(double x, double y);
__device__ static  double __dmul_ru(double x, double y);
__device__ static  double __dmul_rz(double x, double y);
__device__  double __drcp_rd(double x);
__device__  double __drcp_rn(double x);
__device__  double __drcp_ru(double x);
__device__  double __drcp_rz(double x);
__device__  double __dsqrt_rd(double x);
__device__  double __dsqrt_rn(double x);
__device__  double __dsqrt_ru(double x);
__device__  double __dsqrt_rz(double x);
__device__ static  double __dsub_rd(double x, double y);
__device__ static  double __dsub_rn(double x, double y);
__device__ static  double __dsub_ru(double x, double y);
__device__ static  double __dsub_rz(double x, double y);
__device__  double __fma_rd(double x, double y, double z);
__device__  double __fma_rn(double x, double y, double z);
__device__  double __fma_ru(double x, double y, double z);
__device__  double __fma_rz(double x, double y, double z);

// Single Precision Fast Math
extern __attribute__((const)) float __hip_fast_cosf(float) __asm("llvm.cos.f32");
extern __attribute__((const)) float __hip_fast_exp2f(float) __asm("llvm.exp2.f32");
__device__ float __hip_fast_exp10f(float);
__device__ float __hip_fast_expf(float);
__device__ float __hip_fast_frsqrt_rn(float);
extern __attribute__((const)) float __hip_fast_fsqrt_rd(float) __asm("llvm.sqrt.f32");
__device__ float __hip_fast_fsqrt_rn(float);
__device__ float __hip_fast_fsqrt_ru(float);
__device__ float __hip_fast_fsqrt_rz(float);
__device__ float __hip_fast_log10f(float);
extern __attribute__((const)) float __hip_fast_log2f(float) __asm("llvm.log2.f32");
__device__ float __hip_fast_logf(float);
__device__ float __hip_fast_powf(float, float);
__device__ void __hip_fast_sincosf(float,float*,float*);
extern __attribute__((const)) float __hip_fast_sinf(float) __asm("llvm.sin.f32");
__device__ float __hip_fast_tanf(float);
extern __attribute__((const)) float __hip_fast_fmaf(float,float,float) __asm("llvm.fma.f32");
extern __attribute__((const)) float __hip_fast_frcp(float) __asm("llvm.amdgcn.rcp.f32");

extern __attribute__((const)) double __hip_fast_dsqrt(double) __asm("llvm.sqrt.f64");
extern __attribute__((const)) double __hip_fast_fma(double,double,double) __asm("llvm.fma.f64");
extern __attribute__((const)) double __hip_fast_drcp(double) __asm("llvm.amdgcn.rcp.f64");


// Single Precision Fast Math
__device__ inline float __cosf(float x) {
  return __hip_fast_cosf(x);
}

__device__ inline float __exp10f(float x) {
  return __hip_fast_exp10f(x);
}

__device__ inline float __expf(float x) {
  return __hip_fast_expf(x);
}

__device__ static inline float __fadd_rd(float x, float y) {
  return x + y;
}

__device__ static inline float __fadd_rn(float x, float y) {
  return x + y;
}

__device__ static inline float __fadd_ru(float x, float y) {
  return x + y;
}

__device__ static inline float __fadd_rz(float x, float y) {
  return x + y;
}

__device__ static inline float __fdiv_rd(float x, float y) {
  return x / y;
}

__device__ static inline float __fdiv_rn(float x, float y) {
  return x / y;
}

__device__ static inline float __fdiv_ru(float x, float y) {
  return x / y;
}

__device__ static inline float __fdiv_rz(float x, float y) {
  return x / y;
}

__device__ static inline float __fdividef(float x, float y) {
  return x / y;
}

__device__ inline float __fmaf_rd(float x, float y, float z) {
  return __hip_fast_fmaf(x, y, z);
}

__device__ inline float __fmaf_rn(float x, float y, float z) {
  return __hip_fast_fmaf(x, y, z);
}

__device__ inline float __fmaf_ru(float x, float y, float z) {
  return __hip_fast_fmaf(x, y, z);
}

__device__ inline float __fmaf_rz(float x, float y, float z) {
  return __hip_fast_fmaf(x, y, z);
}

__device__ static inline float __fmul_rd(float x, float y) {
  return x * y;
}

__device__ static inline float __fmul_rn(float x, float y) {
  return x * y;
}

__device__ static inline float __fmul_ru(float x, float y) {
  return x * y;
}

__device__ static inline float __fmul_rz(float x, float y) {
  return x * y;
}

__device__ inline float __frcp_rd(float x) {
  return __hip_fast_frcp(x);
}

__device__ inline float __frcp_rn(float x) {
  return __hip_fast_frcp(x);
}

__device__ inline float __frcp_ru(float x) {
  return __hip_fast_frcp(x);
}

__device__ inline float __frcp_rz(float x) {
  return __hip_fast_frcp(x);
}

__device__ inline float __frsqrt_rn(float x) {
  return __hip_fast_frsqrt_rn(x);
}

__device__ inline float __fsqrt_rd(float x) {
  return __hip_fast_fsqrt_rd(x);
}

__device__ inline float __fsqrt_rn(float x) {
  return __hip_fast_fsqrt_rn(x);
}

__device__ inline float __fsqrt_ru(float x) {
  return __hip_fast_fsqrt_ru(x);
}

__device__ inline float __fsqrt_rz(float x) {
  return __hip_fast_fsqrt_rz(x);
}

__device__ static inline float __fsub_rd(float x, float y) {
  return x - y;
}

__device__ static inline float __fsub_rn(float x, float y) {
  return x - y;
}

__device__ static inline float __fsub_ru(float x, float y) {
  return x - y;
}

__device__ static inline float __fsub_rz(float x, float y) {
  return x - y;
}


__device__ inline float __log10f(float x) {
  return __hip_fast_log10f(x);
}

__device__ inline float __log2f(float x) {
  return __hip_fast_log2f(x);
}

__device__ inline float __logf(float x) {
  return __hip_fast_logf(x);
}

__device__ inline float __powf(float base, float exponent) {
  return __hip_fast_powf(base, exponent);
}

__device__ static inline float __saturatef(float x) {
  x = x > 1.0f ? 1.0f : x;
  x = x < 0.0f ? 0.0f : x;
  return x;
}

__device__ inline void __sincosf(float x, float *s, float *c) {
  return __hip_fast_sincosf(x, s, c);
}

__device__ inline float __sinf(float x) {
  return __hip_fast_sinf(x);
}

__device__ inline float __tanf(float x) {
  return __hip_fast_tanf(x);
}


/*
Double Precision Intrinsics
*/

__device__ static inline double __dadd_rd(double x, double y) {
  return x + y;
}

__device__ static inline double __dadd_rn(double x, double y) {
  return x + y;
}

__device__ static inline double __dadd_ru(double x, double y) {
  return x + y;
}

__device__ static inline double __dadd_rz(double x, double y) {
  return x + y;
}

__device__ static inline double __ddiv_rd(double x, double y) {
  return x / y;
}

__device__ static inline double __ddiv_rn(double x, double y) {
  return x / y;
}

__device__ static inline double __ddiv_ru(double x, double y) {
  return x / y;
}

__device__ static inline double __ddiv_rz(double x, double y) {
  return x / y;
}

__device__ static inline double __dmul_rd(double x, double y) {
  return x * y;
}

__device__ static inline double __dmul_rn(double x, double y) {
  return x * y;
}

__device__ static inline double __dmul_ru(double x, double y) {
  return x * y;
}

__device__ static inline double __dmul_rz(double x, double y) {
  return x * y;
}

__device__ inline double __drcp_rd(double x) {
  return __hip_fast_drcp(x);
}

__device__ inline double __drcp_rn(double x) {
  return __hip_fast_drcp(x);
}

__device__ inline double __drcp_ru(double x) {
  return __hip_fast_drcp(x);
}

__device__ inline double __drcp_rz(double x) {
  return __hip_fast_drcp(x);
}


__device__ inline double __dsqrt_rd(double x) {
  return __hip_fast_dsqrt(x);
}

__device__ inline double __dsqrt_rn(double x) {
  return __hip_fast_dsqrt(x);
}

__device__ inline double __dsqrt_ru(double x) {
  return __hip_fast_dsqrt(x);
}

__device__ inline double __dsqrt_rz(double x) {
  return __hip_fast_dsqrt(x);
}

__device__ static inline double __dsub_rd(double x, double y) {
  return x - y;
}

__device__ static inline double __dsub_rn(double x, double y) {
  return x - y;
}

__device__ static inline double __dsub_ru(double x, double y) {
  return x - y;
}

__device__ static inline double __dsub_rz(double x, double y) {
  return x - y;
}

__device__ inline double __fma_rd(double x, double y, double z) {
  return __hip_fast_fma(x, y, z);
}

__device__ inline double __fma_rn(double x, double y, double z) {
  return __hip_fast_fma(x, y, z);
}

__device__ inline double __fma_ru(double x, double y, double z) {
  return __hip_fast_fma(x, y, z);
}

__device__ inline double __fma_rz(double x, double y, double z) {
  return __hip_fast_fma(x, y, z);
}


extern "C" unsigned int __hip_hc_ir_umul24_int(unsigned int, unsigned int);
extern "C" signed int __hip_hc_ir_mul24_int(signed int, signed int);
extern "C" signed int __hip_hc_ir_mulhi_int(signed int, signed int);
extern "C" unsigned int __hip_hc_ir_umulhi_int(unsigned int, unsigned int);
extern "C" unsigned int __hip_hc_ir_usad_int(unsigned int, unsigned int, unsigned int);

// integer intrinsic function __poc __clz __ffs __brev
__device__ unsigned int __brev( unsigned int x);
__device__ unsigned long long int __brevll( unsigned long long int x);
__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
__device__ unsigned int __clz(int x);
__device__ unsigned int __clzll(long long int x);
__device__ unsigned int __ffs(int x);
__device__ unsigned int __ffsll(long long int x);
__device__ static unsigned int __hadd(int x, int y);
__device__ static int __mul24(int x, int y);
__device__ long long int __mul64hi(long long int x, long long int y);
__device__ static int __mulhi(int x, int y);
__device__ unsigned int __popc(unsigned int x);
__device__ unsigned int __popcll(unsigned long long int x);
__device__ static int __rhadd(int x, int y);
__device__ static unsigned int __sad(int x, int y, int z);
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);
__device__ static int __umul24(unsigned int x, unsigned int y);
__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

__device__ static inline unsigned int __hadd(int x, int y) {
  int z = x + y;
  int sign = z & 0x8000000;
  int value = z & 0x7FFFFFFF;
  return ((value) >> 1 || sign);
}
__device__ static inline int __mul24(int x, int y) {
  return __hip_hc_ir_mul24_int(x, y);
}
__device__ static inline int __mulhi(int x, int y) {
  return __hip_hc_ir_mulhi_int(x, y);
}
__device__ static inline int __rhadd(int x, int y) {
  int z = x + y + 1;
  int sign = z & 0x8000000;
  int value = z & 0x7FFFFFFF;
  return ((value) >> 1 || sign);
}
__device__ static inline unsigned int __sad(int x, int y, int z) {
  return x > y ? x - y + z : y - x + z;
}
__device__ static inline unsigned int __uhadd(unsigned int x, unsigned int y) {
  return (x + y) >> 1;
}
__device__ static inline int __umul24(unsigned int x, unsigned int y) {
  return __hip_hc_ir_umul24_int(x, y);
}
__device__ static inline unsigned int __umulhi(unsigned int x, unsigned int y) {
  return __hip_hc_ir_umulhi_int(x, y);
}
__device__ static inline unsigned int __urhadd(unsigned int x, unsigned int y) {
  return (x + y + 1) >> 1;
}
__device__ static inline unsigned int __usad(unsigned int x, unsigned int y, unsigned int z)
{
  return __hip_hc_ir_usad_int(x, y, z);
}

/*
Rounding modes are not yet supported in HIP
*/

__device__ float __double2float_rd(double x);
__device__ float __double2float_rn(double x);
__device__ float __double2float_ru(double x);
__device__ float __double2float_rz(double x);

__device__ int __double2hiint(double x);

__device__ int __double2int_rd(double x);
__device__ int __double2int_rn(double x);
__device__ int __double2int_ru(double x);
__device__ int __double2int_rz(double x);

__device__ long long int __double2ll_rd(double x);
__device__ long long int __double2ll_rn(double x);
__device__ long long int __double2ll_ru(double x);
__device__ long long int __double2ll_rz(double x);

__device__ int __double2loint(double x);

__device__ unsigned int __double2uint_rd(double x);
__device__ unsigned int __double2uint_rn(double x);
__device__ unsigned int __double2uint_ru(double x);
__device__ unsigned int __double2uint_rz(double x);

__device__ unsigned long long int __double2ull_rd(double x);
__device__ unsigned long long int __double2ull_rn(double x);
__device__ unsigned long long int __double2ull_ru(double x);
__device__ unsigned long long int __double2ull_rz(double x);

__device__ long long int __double_as_longlong(double x);
/*
__device__ unsigned short __float2half_rn(float x);
__device__ float __half2float(unsigned short);

The above device function are not a valid .
Use
__device__ __half __float2half_rn(float x);
__device__ float __half2float(__half);
from hip_fp16.h

CUDA implements half as unsigned short whereas, HIP doesn't.

*/

__device__ int __float2int_rd(float x);
__device__ int __float2int_rn(float x);
__device__ int __float2int_ru(float x);
__device__ int __float2int_rz(float x);

__device__ long long int __float2ll_rd(float x);
__device__ long long int __float2ll_rn(float x);
__device__ long long int __float2ll_ru(float x);
__device__ long long int __float2ll_rz(float x);

__device__ unsigned int __float2uint_rd(float x);
__device__ unsigned int __float2uint_rn(float x);
__device__ unsigned int __float2uint_ru(float x);
__device__ unsigned int __float2uint_rz(float x);

__device__ unsigned long long int __float2ull_rd(float x);
__device__ unsigned long long int __float2ull_rn(float x);
__device__ unsigned long long int __float2ull_ru(float x);
__device__ unsigned long long int __float2ull_rz(float x);

__device__ int __float_as_int(float x);
__device__ unsigned int __float_as_uint(float x);
__device__ double __hiloint2double(int hi, int lo);
__device__ double __int2double_rn(int x);

__device__ float __int2float_rd(int x);
__device__ float __int2float_rn(int x);
__device__ float __int2float_ru(int x);
__device__ float __int2float_rz(int x);

__device__ float __int_as_float(int x);

__device__ double __ll2double_rd(long long int x);
__device__ double __ll2double_rn(long long int x);
__device__ double __ll2double_ru(long long int x);
__device__ double __ll2double_rz(long long int x);

__device__ float __ll2float_rd(long long int x);
__device__ float __ll2float_rn(long long int x);
__device__ float __ll2float_ru(long long int x);
__device__ float __ll2float_rz(long long int x);

__device__ double __longlong_as_double(long long int x);

__device__ double __uint2double_rn(int x);

__device__ float __uint2float_rd(unsigned int x);
__device__ float __uint2float_rn(unsigned int x);
__device__ float __uint2float_ru(unsigned int x);
__device__ float __uint2float_rz(unsigned int x);

__device__ float __uint_as_float(unsigned int x);

__device__ double __ull2double_rd(unsigned long long int x);
__device__ double __ull2double_rn(unsigned long long int x);
__device__ double __ull2double_ru(unsigned long long int x);
__device__ double __ull2double_rz(unsigned long long int x);

__device__ float __ull2float_rd(unsigned long long int x);
__device__ float __ull2float_rn(unsigned long long int x);
__device__ float __ull2float_ru(unsigned long long int x);
__device__ float __ull2float_rz(unsigned long long int x);

__device__ char4 __hip_hc_add8pk(char4, char4);
__device__ char4 __hip_hc_sub8pk(char4, char4);
__device__ char4 __hip_hc_mul8pk(char4, char4);


#endif
