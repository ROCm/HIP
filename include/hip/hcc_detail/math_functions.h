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

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_MATH_FUNCTIONS_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_MATH_FUNCTIONS_H

#if defined(__HCC__)
   #include <kalmar_math.h>
#endif


#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hip/hcc_detail/device_functions.h>

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
__device__ float cosf(float x);
__device__ float coshf(float x);
__device__ float cospif(float x);
//__device__ float cyl_bessel_i0f(float x);
//__device__ float cyl_bessel_i1f(float x);
__device__ float erfcf(float x);
__device__  float erfcinvf(float y);
__device__ float erfcxf(float x);
__device__ float erff(float x);
__device__ float erfinvf(float y);
__device__ float exp10f(float x);
__device__ float exp2f(float x);
__device__ float expf(float x);
__device__ float expm1f(float x);
__device__ int abs(int x);
__device__ float fabsf(float x);
__device__ float fdimf(float x, float y);
__device__ float fdividef(float x, float y);
__device__ float floorf(float x);
__device__ float fmaf(float x, float y, float z);
__device__ float fmaxf(float x, float y);
__device__ float fminf(float x, float y);
__device__ float fmodf(float x, float y);
__device__ float frexpf(float x, int* nptr);
__device__ float hypotf(float x, float y);
__device__ float ilogbf(float x);
__device__ int isfinite(float a);
__device__ unsigned isinf(float a);
__device__ unsigned isnan(float a);
__device__ float j0f(float x);
__device__ float j1f(float x);
__device__ float jnf(int n, float x);
__device__ float ldexpf(float x, int exp);
__device__ float lgammaf(float x);
__device__ long long int llrintf(float x);
__device__ long long int llroundf(float x);
__device__ float log10f(float x);
__device__ float log1pf(float x);
__device__ float logbf(float x);
__device__ long int lrintf(float x);
__device__ long int lroundf(float x);
//__device__ float modff(float x, float *iptr);
__device__ float nanf(const char* tagp);
__device__ float nearbyintf(float x);
//__device__ float nextafterf(float x, float y);
__device__ float norm3df(float a, float b, float c);
__device__ float norm4df(float a, float b, float c, float d);
__device__ float normcdff(float y);
__device__ float normcdfinvf(float y);
__device__ float normf(int dim, const float *a);
__device__ float powf(float x, float y);
__device__ float rcbrtf(float x);
__device__ float remainderf(float x, float y);
__device__ float remquof(float x, float y, int *quo);
__device__ float rhypotf(float x, float y);
__device__ float rintf(float x);
__device__ float rnorm3df(float a, float b, float c);
__device__ float rnorm4df(float a, float b, float c, float d);
__device__ float rnormf(int dim, const float* a);
__device__ float roundf(float x);
__device__ float rsqrtf(float x);
__device__ float scalblnf(float x, long int n);
__device__ float scalbnf(float x, int n);
__device__ int signbit(float a);
__device__ void sincosf(float x, float *sptr, float *cptr);
__device__ void sincospif(float x, float *sptr, float *cptr);
__device__ float sinf(float x);
__device__ float sinhf(float x);
__device__ float sinpif(float x);
__device__ float sqrtf(float x);
__device__ float tanf(float x);
__device__ float tanhf(float x);
__device__ float tgammaf(float x);
__device__ float truncf(float x);
__device__ float y0f(float x);
__device__ float y1f(float x);
__device__ float ynf(int n, float x);

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
__device__ double cospi(double x);
//__device__ double cyl_bessel_i0(double x);
//__device__ double cyl_bessel_i1(double x);
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
__device__ double floor(double x);
__device__ double fma(double x, double y, double z);
__device__ double fmax(double x, double y);
__device__ double fmin(double x, double y);
__device__ double fmod(double x, double y);
__device__ double frexp(double x, int *nptr);
__device__ double hypot(double x, double y);
__device__ double ilogb(double x);
__device__ int isfinite(double x);
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
//__device__ double modf(double x, double *iptr);
__device__ double nan(const char* tagp);
__device__ double nearbyint(double x);
__device__ double nextafter(double x, double y);
__device__ double norm(int dim, const double* t);
__device__ double norm3d(double a, double b, double c);
__device__ double norm4d(double a, double b, double c, double d);
__device__ double normcdf(double y);
__device__ double normcdfinv(double y);
__device__ double pow(double x, double y);
__device__ double rcbrt(double x);
__device__ double remainder(double x, double y);
//__device__ double remquo(double x, double y, int *quo);
__device__ double rhypot(double x, double y);
__device__ double rint(double x);
__device__ double rnorm(int dim, const double* t);
__device__ double rnorm3d(double a, double b, double c);
__device__ double rnorm4d(double a, double b, double c, double d);
__device__ double round(double x);
__device__ double rsqrt(double x);
__device__ double scalbln(double x, long int n);
__device__ double scalbn(double x, int n);
__device__ int signbit(double a);
__device__ double sin(double a);
__device__ void sincos(double x, double *sptr, double *cptr);
__device__ void sincospi(double x, double *sptr, double *cptr);
__device__ double sinh(double x);
__device__ double sinpi(double x);
__device__ double sqrt(double x);
__device__ double tan(double x);
__device__ double tanh(double x);
__device__ double tgamma(double x);
__device__ double trunc(double x);
__device__ double y0(double x);
__device__ double y1(double y);
__device__ double yn(int n, double x);

// ENDPARSER

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


#endif
