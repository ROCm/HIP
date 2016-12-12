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
 *  @file  hcc_detail/hip_runtime.h
 *  @brief Contains definitions of APIs for HIP runtime.
 */

//#pragma once
#ifndef HIP_RUNTIME_H
#define HIP_RUNTIME_H

//---
// Top part of file can be compiled with any compiler


//#include <cstring>
#if __cplusplus
#include <cmath>
#else
#include <math.h>
#include <string.h>
#include <stddef.h>
#endif
// Define NVCC_COMPAT for CUDA compatibility
#define NVCC_COMPAT
#define CUDA_SUCCESS hipSuccess

#include <hip/hip_runtime_api.h>
//#include "hip/hcc_detail/hip_hcc.h"
//---
// Remainder of this file only compiles with HCC
#ifdef __HCC__
#include <grid_launch.h>

#if defined (GRID_LAUNCH_VERSION) and (GRID_LAUNCH_VERSION >= 20)
// Use field names for grid_launch 2.0 structure, if HCC supports GL 2.0.
#else
#error (HCC must support GRID_LAUNCH_20)
#endif

extern int HIP_TRACE_API;

//TODO-HCC-GL - change this to typedef.
//typedef grid_launch_parm hipLaunchParm ;
#define hipLaunchParm grid_launch_parm
#ifdef __cplusplus
//#include <hip/hcc_detail/hip_texture.h>
#include <hip/hcc_detail/hip_ldg.h>
#endif
#include <hip/hcc_detail/host_defines.h>
// TODO-HCC remove old definitions ; ~1602 hcc supports __HCC_ACCELERATOR__ define.
#if defined (__KALMAR_ACCELERATOR__) && !defined (__HCC_ACCELERATOR__)
#define __HCC_ACCELERATOR__  __KALMAR_ACCELERATOR__
#endif

// Feature tests:
#if defined(__HCC_ACCELERATOR__) && (__HCC_ACCELERATOR__ != 0)
// Device compile and not host compile:

//TODO-HCC enable __HIP_ARCH_HAS_ATOMICS__ when HCC supports these.
    // 32-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__       (1)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__   (1)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__       (1)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__   (1)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__           (0)

// 64-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__       (1)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__       (0)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__                    (1)

//warp cross-lane operations:
#define __HIP_ARCH_HAS_WARP_VOTE__                  (1)
#define __HIP_ARCH_HAS_WARP_BALLOT__                (1)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__               (1)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__          (0)

//sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__        (0)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__            (0)

// misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__              (0)
#define __HIP_ARCH_HAS_3DGRID__                     (1)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__           (0)

#endif /* Device feature flags */


//TODO-HCC  this is currently ignored by HCC target of HIP
#define __launch_bounds__(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)

// Detect if we are compiling C++ mode or C mode
#if defined(__cplusplus)
#define __HCC_CPP__
#elif defined(__STDC_VERSION__)
#define __HCC_C__
#endif

__device__ float acosf(float x);
__device__ float acoshf(float x);
__device__ float asinf(float x);
__device__ float asinhf(float x);
__device__ float atan2f(float y, float x);
__device__ float atanf(float x);
__device__ float atanhf(float x);
__device__ float cbrtf(float x);
__device__ float ceilf(float x);
__device__ float copysignf(float x, float y);
__device__ float coshf(float x);
__device__ float cyl_bessel_i0f(float x);
__device__ float cyl_bessel_i1f(float x);
__device__ float erfcf(float x);
__device__  float erfcinvf(float y);
__host__ float erfcinvf(float y);
__device__ float erfcxf(float x);
__host__ float erfcxf(float x);
__device__ float erff(float x);
__device__ float erfinvf(float y);
__host__ float erfinvf(float y);
__device__ float exp2f(float x);
__device__ float expm1f(float x);
__device__ float fabsf(float x);
__device__ float fdimf(float x, float y);
__device__ __host__ float fdividef(float x, float y);
__device__ float floorf(float x);
__device__ float fmaf(float x, float y, float z);
__device__ float fmaxf(float x, float y);
__device__ float fminf(float x, float y);
__device__ float fmodf(float x, float y);
__device__ float frexpf(float x, float y);
__device__ float hypotf(float x, float y);
__device__ float ilogbf(float x);
__host__ __device__ unsigned isfinite(float a);
__device__ unsigned isinf(float a);
__device__ unsigned isnan(float a);
__device__ float j0f(float x);
__device__ float j1f(float x);
__device__ float jnf(int n, float x);
__device__ float ldexpf(float x, int exp);
__device__ float lgammaf(float x);
__device__ long long int llrintf(float x);
__device__ long long int llroundf(float x);
__device__ float log1pf(float x);
__device__ float logbf(float x);
__device__ long int lrintf(float x);
__device__ long int lroundf(float x);
__device__ float modff(float x, float *iptr);
__device__ float nanf(const char* tagp);
__device__ float nearbyintf(float x);
__device__ float nextafterf(float x, float y);
__device__ float norm3df(float a, float b, float c);
__host__ float norm3df(float a, float b, float c);
__device__ float norm4df(float a, float b, float c, float d);
__host__ float norm4df(float a, float b, float c, float d);
__device__ float normcdff(float y);
__host__ float normcdff(float y);
__device__ float normcdfinvf(float y);
__host__ float normcdfinvf(float y);
__device__ float normf(int dim, const float *a);
__device__ float rcbrtf(float x);
__host__ float rcbrtf(float x);
__device__ float remainderf(float x, float y);
__device__ float remquof(float x, float y, int *quo);
__device__ float rhypotf(float x, float y);
__host__ float rhypotf(float x, float y);
__device__ float rintf(float x);
__device__ float rnorm3df(float a, float b, float c);
__host__ float rnorm3df(float a, float b, float c);
__device__ float rnorm4df(float a, float b, float c, float d);
__host__ float rnorm4df(float a, float b, float c, float d);
__device__ float rnormf(int dim, const float* a);
__host__ float rnormf(int dim, const float* a);
__device__ float roundf(float x);
__device__ float rsqrtf(float x);
__device__ float scalblnf(float x, long int n);
__device__ float scalbnf(float x, int n);
__host__ __device__ unsigned signbit(float a);
__device__ void sincospif(float x, float *sptr, float *cptr);
__host__ void sincospif(float x, float *sptr, float *cptr);
__device__ float sinhf(float x);
__device__ float sinpif(float x);
__device__ float sqrtf(float x);
__device__ float tanhf(float x);
__device__ float tgammaf(float x);
__device__ float truncf(float x);
__device__ float y0f(float x);
__device__ float y1f(float x);
__device__ float ynf(int n, float x);

__host__ __device__ float cospif(float x);
__host__ __device__ float sinpif(float x);
__device__ float sqrtf(float x);
__host__ __device__ float rsqrtf(float x);

__device__ double acos(double x);
__device__ double acosh(double x);
__device__ double asin(double x);
__device__ double asinh(double x);
__device__ double atan(double x);
__device__ double atan2(double y, double x);
__device__ double atanh(double x);
__device__ double cbrt(double x);
__device__ double ceil(double x);
__device__ double copysign(double x, double y);
__device__ double cos(double x);
__device__ double cosh(double x);
__host__ __device__ double cospi(double x);
__device__ double cyl_bessel_i0(double x);
__device__ double cyl_bessel_i1(double x);
__device__ double erf(double x);
__device__ double erfc(double x);
__device__ double erfcinv(double y);
__device__ double erfcx(double x);
__device__ double erfinv(double x);
__device__ double exp(double x);
__device__ double exp10(double x);
__device__ double exp2(double x);
__device__ double expm1(double x);
__device__ double fabs(double x);
__device__ double fdim(double x, double y);
__device__ double fdivide(double x, double y);
__device__ double floor(double x);
__device__ double fma(double x, double y, double z);
__device__ double fmax(double x, double y);
__device__ double fmin(double x, double y);
__device__ double fmod(double x, double y);
__device__ double frexp(double x, int *nptr);
__device__ double hypot(double x, double y);
__device__ double ilogb(double x);
__host__ __device__ unsigned isfinite(double x);
__device__ unsigned isinf(double x);
__device__ unsigned isnan(double x);
__device__ double j0(double x);
__device__ double j1(double x);
__device__ double jn(int n, double x);
__device__ double ldexp(double x, int exp);
__device__ double lgamma(double x);
__device__ long long llrint(double x);
__device__ long long llround(double x);
__device__ double log(double x);
__device__ double log10(double x);
__device__ double log1p(double x);
__device__ double log2(double x);
__device__ double logb(double x);
__device__ long int lrint(double x);
__device__ long int lround(double x);
__device__ double modf(double x, double *iptr);
__device__ double nan(const char* tagp);
__device__ double nearbyint(double x);
__device__ double nextafter(double x, double y);
__device__ double norm(int dim, const double* t);
__device__ double norm3d(double a, double b, double c);
__host__ double norm3d(double a, double b, double c);
__device__ double norm4d(double a, double b, double c, double d);
__host__ double norm4d(double a, double b, double c, double d);
__device__ double normcdf(double y);
__host__ double normcdf(double y);
__device__ double normcdfinv(double y);
__host__ double normcdfinv(double y);
__device__ double pow(double x, double y);
__device__ double rcbrt(double x);
__host__ double rcbrt(double x);
__device__ double remainder(double x, double y);
__device__ double remquo(double x, double y, int *quo);
__device__ double rhypot(double x, double y);
__host__ double rhypot(double x, double y);
__device__ double rint(double x);
__device__ double rnorm(int dim, const double* t);
__host__ double rnorm(int dim, const double* t);
__device__ double rnorm3d(double a, double b, double c);
__host__ double rnorm3d(double a, double b, double c);
__device__ double rnorm4d(double a, double b, double c, double d);
__host__ double rnorm4d(double a, double b, double c, double d);
__device__ double round(double x);
__host__ __device__ double rsqrt(double x);
__device__ double scalbln(double x, long int n);
__device__ double scalbn(double x, int n);
__host__ __device__ unsigned signbit(double a);
__device__ double sin(double a);
__device__ void sincos(double x, double *sptr, double *cptr);
__device__ void sincospi(double x, double *sptr, double *cptr);
__host__ void sincospi(double x, double *sptr, double *cptr);
__device__ double sinh(double x);
__host__ __device__ double sinpi(double x);
__device__ double sqrt(double x);
__device__ double tan(double x);
__device__ double tanh(double x);
__device__ double tgamma(double x);
__device__ double trunc(double x);
__device__ double y0(double x);
__device__ double y1(double y);
__device__ double yn(int n, double x);

__host__ double erfcinv(double y);
__host__ double erfcx(double x);
__host__ double erfinv(double y);
__host__ double fdivide(double x, double y);

// TODO - hipify-clang - change to use the function call.
//#define warpSize hc::__wavesize()
extern const int warpSize;


#define clock_t long long int
__device__ long long int clock64();
__device__ clock_t clock();

//atomicAdd()
__device__ int atomicAdd(int* address, int val);
__device__ unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);

__device__ unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);

__device__ float atomicAdd(float* address, float val);


//atomicSub()
__device__ int atomicSub(int* address, int val);

__device__ unsigned int atomicSub(unsigned int* address,
                       unsigned int val);


//atomicExch()
__device__ int atomicExch(int* address, int val);

__device__ unsigned int atomicExch(unsigned int* address,
                        unsigned int val);

__device__ unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);

__device__ float atomicExch(float* address, float val);


//atomicMin()
__device__ int atomicMin(int* address, int val);
__device__ unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);


//atomicMax()
__device__ int atomicMax(int* address, int val);
__device__ unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);


//atomicCAS()
__device__ int atomicCAS(int* address, int compare, int val);
__device__ unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
__device__ unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);


//atomicAnd()
__device__ int atomicAnd(int* address, int val);
__device__ unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);


//atomicOr()
__device__ int atomicOr(int* address, int val);
__device__ unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
__device__ unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);


//atomicXor()
__device__ int atomicXor(int* address, int val);
__device__ unsigned int atomicXor(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val);

//atomicInc()
__device__ unsigned int atomicInc(unsigned int* address,
                       unsigned int val);


//atomicDec()
__device__ unsigned int atomicDec(unsigned int* address,
                       unsigned int val);

//__mul24 __umul24
__device__  int __mul24(int arg1, int arg2);
__device__  unsigned int __umul24(unsigned int arg1, unsigned int arg2);

// integer intrinsic function __poc __clz __ffs __brev
__device__ unsigned int __popc( unsigned int input);
__device__ unsigned int __popcll( unsigned long long int input);
__device__ unsigned int __clz(unsigned int input);
__device__ unsigned int __clzll(unsigned long long int input);
__device__ unsigned int __clz(int input);
__device__ unsigned int __clzll(long long int input);
__device__ unsigned int __ffs(unsigned int input);
__device__ unsigned int __ffsll(unsigned long long int input);
__device__ unsigned int __ffs(int input);
__device__ unsigned int __ffsll(long long int input);
__device__ unsigned int __brev( unsigned int input);
__device__ unsigned long long int __brevll( unsigned long long int input);


// warp vote function __all __any __ballot
__device__ int __all(  int input);
__device__ int __any( int input);
__device__  unsigned long long int __ballot( int input);

// warp shuffle functions
#ifdef __cplusplus
__device__ int __shfl(int input, int lane, int width=warpSize);
__device__ int __shfl_up(int input, unsigned int lane_delta, int width=warpSize);
__device__ int __shfl_down(int input, unsigned int lane_delta, int width=warpSize);
__device__ int __shfl_xor(int input, int lane_mask, int width=warpSize);
__device__ float __shfl(float input, int lane, int width=warpSize);
__device__ float __shfl_up(float input, unsigned int lane_delta, int width=warpSize);
__device__ float __shfl_down(float input, unsigned int lane_delta, int width=warpSize);
__device__ float __shfl_xor(float input, int lane_mask, int width=warpSize);
#else
__device__ int __shfl(int input, int lane, int width);
__device__ int __shfl_up(int input, unsigned int lane_delta, int width);
__device__ int __shfl_down(int input, unsigned int lane_delta, int width);
__device__ int __shfl_xor(int input, int lane_mask, int width);
__device__ float __shfl(float input, int lane, int width);
__device__ float __shfl_up(float input, unsigned int lane_delta, int width);
__device__ float __shfl_down(float input, unsigned int lane_delta, int width);
__device__ float __shfl_xor(float input, int lane_mask, int width);
#endif

__host__ __device__ int min(int arg1, int arg2);
__host__ __device__ int max(int arg1, int arg2);

__device__ __attribute__((address_space(3))) void* __get_dynamicgroupbaseptr();

//TODO - add a couple fast math operations here, the set here will grow :

// Single Precision Precise Math
__device__ float __hip_precise_cosf(float);
__device__ float __hip_precise_exp10f(float);
__device__ float __hip_precise_expf(float);
__device__ float __hip_precise_frsqrt_rn(float);
__device__ float __hip_precise_fsqrt_rd(float);
__device__ float __hip_precise_fsqrt_rn(float);
__device__ float __hip_precise_fsqrt_ru(float);
__device__ float __hip_precise_fsqrt_rz(float);
__device__ float __hip_precise_log10f(float);
__device__ float __hip_precise_log2f(float);
__device__ float __hip_precise_logf(float);
__device__ float __hip_precise_powf(float, float);
__device__ void __hip_precise_sincosf(float,float*,float*);
__device__ float __hip_precise_sinf(float);
__device__ float __hip_precise_tanf(float);

// Double Precision Precise Math
__device__ double __hip_precise_dsqrt_rd(double);
__device__ double __hip_precise_dsqrt_rn(double);
__device__ double __hip_precise_dsqrt_ru(double);
__device__ double __hip_precise_dsqrt_rz(double);

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

#ifdef HIP_FAST_MATH
// Single Precision Precise Math when enabled

__device__ inline float cosf(float x) {
  return __hip_fast_cosf(x);
}

__device__ inline float exp10f(float x) {
  return __hip_fast_exp10f(x);
}

__device__ inline float expf(float x) {
  return __hip_fast_expf(x);
}

__device__ inline float log10f(float x) {
  return __hip_fast_log10f(x);
}

__device__ inline float log2f(float x) {
  return __hip_fast_log2f(x);
}

__device__ inline float logf(float x) {
  return __hip_fast_logf(x);
}

__device__ inline float powf(float base, float exponent) {
  return __hip_fast_powf(base, exponent);
}

__device__ inline void sincosf(float x, float *s, float *c) {
  return __hip_fast_sincosf(x, s, c);
}

__device__ inline float sinf(float x) {
  return __hip_fast_sinf(x);
}

__device__ inline float tanf(float x) {
  return __hip_fast_tanf(x);
}

#else

__device__ float sinf(float);
__device__ float cosf(float);
__device__ float tanf(float);
__device__ void sincosf(float, float*, float*);
__device__ float logf(float);
__device__ float log2f(float);
__device__ float log10f(float);
__device__ float expf(float);
__device__ float exp10f(float);
__device__ float powf(float, float);

#endif
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

__device__ inline void __sincosf(float x, float *s, float *c) {
  return __hip_fast_sincosf(x, s, c);
}

__device__ inline float __sinf(float x) {
  return __hip_fast_sinf(x);
}

__device__ inline float __tanf(float x) {
  return __hip_fast_tanf(x);
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

/**
 * CUDA 8 device function features

 */


/**
 * Kernel launching
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Fence Fence Functions
 *  @{
 *
 *
 *  @warning The HIP memory fence functions are currently not supported yet.
 *  If any of those threadfence stubs are reached by the application, you should set "export HSA_DISABLE_CACHE=1" to disable L1 and L2 caches.
 *
 *
 *  On AMD platforms, the threadfence* routines are currently empty stubs.
 */

 /**
 * @brief threadfence_block makes writes visible to threads running in same block.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence_block is a stub and map to no-op.
 */
// __device__ void  __threadfence_block(void);
extern "C" __device__ void __threadfence_block(void);

 /**
  * @brief threadfence makes wirtes visible to other threads running on same GPU.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence is a stub and map to no-op, application should set "export HSA_DISABLE_CACHE=1" to disable both L1 and L2 caches.
 */
// __device__ void  __threadfence(void) __attribute__((deprecated("Provided for compile-time compatibility, not yet functional")));
extern "C" __device__ void __threadfence(void);

/**
 * @brief threadfence_system makes writes to pinned system memory visible on host CPU.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence_system is a stub and map to no-op.
 */
//__device__ void  __threadfence_system(void) __attribute__((deprecated("Provided with workaround configuration, see hip_kernel_language.md for details")));
__device__ void  __threadfence_system(void) ;

__device__ unsigned __hip_ds_bpermute(int index, unsigned src);
__device__ float __hip_ds_bpermutef(int index, float src);
__device__ unsigned __hip_ds_permute(int index, unsigned src);
__device__ float __hip_ds_permutef(int index, float src);

__device__ unsigned __hip_ds_swizzle(unsigned int src, int pattern);
__device__ float __hip_ds_swizzlef(float src, int pattern);

__device__ int __hip_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl);

// doxygen end Fence Fence
/**
 * @}
 */


#define hipThreadIdx_x (hc_get_workitem_id(0))
#define hipThreadIdx_y (hc_get_workitem_id(1))
#define hipThreadIdx_z (hc_get_workitem_id(2))

#define hipBlockIdx_x  (hc_get_group_id(0))
#define hipBlockIdx_y  (hc_get_group_id(1))
#define hipBlockIdx_z  (hc_get_group_id(2))

#define hipBlockDim_x  (hc_get_group_size(0))
#define hipBlockDim_y  (hc_get_group_size(1))
#define hipBlockDim_z  (hc_get_group_size(2))

#define hipGridDim_x   (hc_get_num_groups(0))
#define hipGridDim_y   (hc_get_num_groups(1))
#define hipGridDim_z   (hc_get_num_groups(2))

//extern "C" __device__ void* memcpy(void* dst, void* src, size_t size);
//extern "C" __device__ void* memset(void* ptr, uint8_t val, size_t size);

extern "C" __device__ void* __hip_hc_malloc(size_t);
extern "C" __device__ void* __hip_hc_free(void *ptr);

//extern "C" __device__ void* malloc(size_t size);
//extern "C" __device__ void* free(void *ptr);

extern "C" __device__ char4 __hip_hc_add8pk(char4, char4);
extern "C" __device__ char4 __hip_hc_sub8pk(char4, char4);
extern "C" __device__ char4 __hip_hc_mul8pk(char4, char4);

#define __syncthreads() hc_barrier(CLK_LOCAL_MEM_FENCE)

#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define HIP_SYMBOL(X) #X

#ifdef __HCC_CPP__
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, dim3 block, grid_launch_parm *lp, const char *kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, size_t block, grid_launch_parm *lp, const char *kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, dim3 block, grid_launch_parm *lp, const char *kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, size_t block, grid_launch_parm *lp, const char *kernelNameStr);
extern void ihipPostLaunchKernel(const char *kernelName, hipStream_t stream, grid_launch_parm &lp);


// Due to multiple overloaded versions of ihipPreLaunchKernel, the numBlocks3D and blockDim3D can be either size_t or dim3 types
#define hipLaunchKernel(_kernelName, _numBlocks3D, _blockDim3D, _groupMemBytes, _stream, ...) \
do {\
  grid_launch_parm lp;\
  lp.dynamic_group_mem_bytes = _groupMemBytes; \
  hipStream_t trueStream = (ihipPreLaunchKernel(_stream, _numBlocks3D, _blockDim3D, &lp, #_kernelName)); \
  _kernelName (lp, ##__VA_ARGS__);\
  ihipPostLaunchKernel(#_kernelName, trueStream, lp);\
} while(0)


#elif defined (__HCC_C__)

//TODO - develop C interface.

#endif

/**
 * extern __shared__
 */

// Macro to replace extern __shared__ declarations
// to local variable definitions
#define HIP_DYNAMIC_SHARED(type, var) \
    __attribute__((address_space(3))) type* var = \
    (__attribute__((address_space(3))) type*)__get_dynamicgroupbaseptr(); \

#define HIP_DYNAMIC_SHARED_ATTRIBUTE __attribute__((address_space(3)))

#endif // __HCC__


/**
 * @defgroup HIP-ENV HIP Environment Variables
 * @{
 */
//extern int HIP_PRINT_ENV ;   ///< Print all HIP-related environment variables.
//extern int HIP_TRACE_API;    ///< Trace HIP APIs.
//extern int HIP_LAUNCH_BLOCKING ; ///< Make all HIP APIs host-synchronous

/**
 * @}
 */


// End doxygen API:
/**
 *   @}
 */




#endif
