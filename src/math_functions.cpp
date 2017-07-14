
/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hc.hpp>
#include <grid_launch.h>
#include <hc_math.hpp>
#include "device_util.h"
#include "hip/hcc_detail/device_functions.h"
#include "hip/hip_runtime.h"

__device__ float acosf(float x)
{
    return hc::precise_math::acosf(x);
}
__device__ float acoshf(float x)
{
    return hc::precise_math::acoshf(x);
}
__device__ float asinf(float x)
{
    return hc::precise_math::asinf(x);
}
__device__ float asinhf(float x)
{
    return hc::precise_math::asinhf(x);
}
__device__ float atan2f(float y, float x)
{
    return hc::precise_math::atan2f(y, x);
}
__device__ float atanf(float x)
{
    return hc::precise_math::atanf(x);
}
__device__ float atanhf(float x)
{
    return hc::precise_math::atanhf(x);
}
__device__ float cbrtf(float x)
{
    return hc::precise_math::cbrtf(x);
}
__device__ float ceilf(float x)
{
    return hc::precise_math::ceilf(x);
}
__device__ float copysignf(float x, float y)
{
    return hc::precise_math::copysignf(x, y);
}
__device__ float cosf(float x)
{
    return hc::precise_math::cosf(x);
}
__device__ float coshf(float x)
{
    return hc::precise_math::coshf(x);
}
__device__ float cyl_bessel_i0f(float x);
__device__ float cyl_bessel_i1f(float x);
__device__ float erfcf(float x)
{
    return hc::precise_math::erfcf(x);
}
__device__ float erfcinvf(float y)
{
    return __hip_erfinvf(1 - y);
}
__device__ float erfcxf(float x)
{
    return hc::precise_math::expf(x*x)*hc::precise_math::erfcf(x);
}
__device__ float erff(float x)
{
    return hc::precise_math::erff(x);
}
__device__ float erfinvf(float y)
{
    return __hip_erfinvf(y);
}
__device__ float exp10f(float x)
{
    return hc::precise_math::exp10f(x);
}
__device__ float exp2f(float x)
{
    return hc::precise_math::exp2f(x);
}
__device__ float expf(float x)
{
    return hc::precise_math::expf(x);
}
__device__ float expm1f(float x)
{
    return hc::precise_math::expm1f(x);
}
__device__ int abs(int x)
{
    return x >= 0 ? x : -x; // TODO - optimize with OCML
}
__device__ float fabsf(float x)
{
    return hc::precise_math::fabsf(x);
}
__device__ float fdimf(float x, float y)
{
    return hc::precise_math::fdimf(x, y);
}
__device__ float fdividef(float x, float y)
{
    return x/y;
}
__device__ float floorf(float x)
{
    return hc::precise_math::floorf(x);
}
__device__ float fmaf(float x, float y, float z)
{
    return hc::precise_math::fmaf(x, y, z);
}
__device__ float fmaxf(float x, float y)
{
    return hc::precise_math::fmaxf(x, y);
}
__device__ float fminf(float x, float y)
{
    return hc::precise_math::fminf(x, y);
}
__device__ float fmodf(float x, float y)
{
    return hc::precise_math::fmodf(x, y);
}
__device__ float frexpf(float x, int *nptr)
{
    return hc::precise_math::frexpf(x, nptr);
}
__device__ float hypotf(float x, float y)
{
    return hc::precise_math::hypotf(x, y);
}
__device__ float ilogbf(float x)
{
    return hc::precise_math::ilogbf(x);
}
__device__ int isfinite(float a)
{
    return hc::precise_math::isfinite(a);
}
__device__ unsigned isinf(float a)
{
    return hc::precise_math::isinf(a);
}
__device__ unsigned isnan(float a)
{
    return hc::precise_math::isnan(a);
}
__device__ float j0f(float x)
{
    return __hip_j0f(x);
}
__device__ float j1f(float x)
{
    return __hip_j1f(x);
}
__device__ float jnf(int n, float x)
{
    return __hip_jnf(n, x);
}
__device__ float ldexpf(float x, int exp)
{
    return hc::precise_math::ldexpf(x, exp);
}
__device__ float lgammaf(float x)
{
  float val = 0.0f;
  float y = x - 1;
  while(y > 0){
    val += logf(y--);
  }
  return val;
}
__device__ long long int llrintf(float x)
{
    int y = hc::precise_math::roundf(x);
    long long int z = y;
    return z;
}
__device__ long long int llroundf(float x)
{
    int y = hc::precise_math::roundf(x);
    long long int z = y;
    return z;
}
__device__ float log10f(float x)
{
    return hc::precise_math::log10f(x);
}
__device__ float log1pf(float x)
{
    return hc::precise_math::log1pf(x);
}
__device__ float log2f(float x)
{
    return hc::precise_math::log2f(x);
}
__device__ float logbf(float x)
{
    return hc::precise_math::logbf(x);
}
__device__ float logf(float x)
{
    return hc::precise_math::logf(x);
}
__device__ long int lrintf(float x)
{
    int y = hc::precise_math::roundf(x);
    long int z = y;
    return z;
}
__device__ long int lroundf(float x)
{
    long int y = hc::precise_math::roundf(x);
    return y;
}
__device__ float modff(float x, float *iptr)
{
    return hc::precise_math::modff(x, iptr);
}
__device__ float nanf(const char* tagp)
{
    return hc::precise_math::nanf((int)*tagp);
}
__device__ float nearbyintf(float x)
{
    return hc::precise_math::nearbyintf(x);
}
__device__ float nextafterf(float x, float y)
{
    return hc::precise_math::nextafter(x, y);
}
__device__ float norm3df(float a, float b, float c)
{
     float x = a*a + b*b + c*c;
     return hc::precise_math::sqrtf(x);
}
__device__ float norm4df(float a, float b, float c, float d)
{
     float x = a*a + b*b;
     float y = c*c + d*d;
     return hc::precise_math::sqrtf(x+y);
}

__device__ float normcdff(float y)
{
     return ((hc::precise_math::erff(y)/1.41421356237) + 1)/2;
}
__device__ float normcdfinvf(float y)
{
     return HIP_SQRT_2 * __hip_erfinvf(2*y-1);
}
__device__ float normf(int dim, const float *a)
{
    float x = 0.0f;
    for(int i=0;i<dim;i++)
    {
        x = hc::precise_math::fmaf(a[i], a[i], x);
    }
    return hc::precise_math::sqrtf(x);
}
__device__ float powf(float x, float y)
{
    return hc::precise_math::powf(x, y);
}
__device__ float rcbrtf(float x)
{
    return hc::precise_math::rcbrtf(x);
}
__device__ float remainderf(float x, float y)
{
    return hc::precise_math::remainderf(x, y);
}
__device__ float remquof(float x, float y, int *quo)
{
    return hc::precise_math::remquof(x, y, quo);
}
__device__ float rhypotf(float x, float y)
{
    return 1/hc::precise_math::hypotf(x, y);
}
__device__ float rintf(float x)
{
    return hc::precise_math::roundf(x);
}
__device__ float rnorm3df(float a, float b, float c)
{
    float x = a*a + b*b + c*c;
    return 1/hc::precise_math::sqrtf(x);
}
__device__ float rnorm4df(float a, float b, float c, float d)
{
    float x = a*a + b*b;
    float y = c*c + d*d;
    return 1/hc::precise_math::sqrtf(x+y);
}
__device__ float rnormf(int dim, const float* a)
{
    float x = 0.0f;
    for(int i=0;i<dim;i++)
    {
        x = hc::precise_math::fmaf(a[i], a[i], x);
    }
    return 1/hc::precise_math::sqrtf(x);
}
__device__ float roundf(float x)
{
    return hc::precise_math::roundf(x);
}
__device__ float scalblnf(float x, long int n)
{
    return hc::precise_math::scalb(x, n);
}
__device__ float scalbnf(float x, int n)
{
    return hc::precise_math::scalbnf(x, n);
}
__device__ int signbit(float a)
{
    return hc::precise_math::signbit(a);
}
__device__ void sincosf(float x, float *sptr, float *cptr)
{
    *sptr = hc::precise_math::sinf(x);
    *cptr = hc::precise_math::cosf(x);
}
__device__ void sincospif(float x, float *sptr, float *cptr)
{
    *sptr = hc::precise_math::sinpif(x);
    *cptr = hc::precise_math::cospif(x);
}
__device__ float sinf(float x)
{
    return hc::precise_math::sinf(x);
}
__device__ float sinhf(float x)
{
    return hc::precise_math::sinhf(x);
}
__device__ float tanf(float x)
{
    return hc::precise_math::tanf(x);
}
__device__ float tanhf(float x)
{
    return hc::precise_math::tanhf(x);
}
__device__ float tgammaf(float x)
{
    return hc::precise_math::tgammaf(x);
}
__device__ float truncf(float x)
{
    return hc::precise_math::truncf(x);
}
__device__ float y0f(float x)
{
    return __hip_y0f(x);
}
__device__ float y1f(float x)
{
    return __hip_y1f(x);
}
__device__ float ynf(int n, float x)
{
    return __hip_ynf(n, x);
}
__device__ float cospif(float x)
{
    return hc::precise_math::cospif(x);
}
__device__ float sinpif(float x)
{
    return hc::precise_math::sinpif(x);
}
__device__ float sqrtf(float x)
{
    return hc::precise_math::sqrtf(x);
}
__device__ float rsqrtf(float x)
{
    return hc::precise_math::rsqrtf(x);
}

/*
 * Double precision device math functions
 */

__device__ double acos(double x)
{
    return hc::precise_math::acos(x);
}
__device__ double acosh(double x)
{
    return hc::precise_math::acosh(x);
}
__device__ double asin(double x)
{
    return hc::precise_math::asin(x);
}
__device__ double asinh(double x)
{
    return hc::precise_math::asinh(x);
}
__device__ double atan(double x)
{
    return hc::precise_math::atan(x);
}
__device__ double atan2(double y, double x)
{
    return hc::precise_math::atan2(y, x);
}
__device__ double atanh(double x)
{
    return hc::precise_math::atanh(x);
}
__device__ double cbrt(double x)
{
    return hc::precise_math::cbrt(x);
}
__device__ double ceil(double x)
{
    return hc::precise_math::ceil(x);
}
__device__ double copysign(double x, double y)
{
    return hc::precise_math::copysign(x, y);
}
__device__ double cos(double x)
{
    return hc::precise_math::cos(x);
}
__device__ double cosh(double x)
{
    return hc::precise_math::cosh(x);
}
__device__ double cospi(double x)
{
    return hc::precise_math::cospi(x);
}
__device__ double cyl_bessel_i0(double x);
__device__ double cyl_bessel_i1(double x);
__device__ double erf(double x)
{
    return hc::precise_math::erf(x);
}
__device__ double erfc(double x)
{
    return hc::precise_math::erfc(x);
}
__device__ double erfcinv(double x)
{
    return __hip_erfinv(1 - x);
}
__device__ double erfcx(double x)
{
    return hc::precise_math::exp(x*x)*hc::precise_math::erf(x);
}
__device__ double erfinv(double x)
{
    return __hip_erfinv(x);
}
__device__ double exp(double x)
{
    return hc::precise_math::exp(x);
}
__device__ double exp10(double x)
{
    return hc::precise_math::exp10(x);
}
__device__ double exp2(double x)
{
    return hc::precise_math::exp2(x);
}
__device__ double expm1(double x)
{
    return hc::precise_math::expm1(x);
}
__device__ double fabs(double x)
{
    return hc::precise_math::fabs(x);
}
__device__ double fdim(double x, double y)
{
    return hc::precise_math::fdim(x, y);
}
__device__ double fdivide(double x, double y)
{
    return x/y;
}
__device__ double floor(double x)
{
    return hc::precise_math::floor(x);
}
__device__ double fma(double x, double y, double z)
{
    return hc::precise_math::fma(x, y, z);
}
__device__ double fmax(double x, double y)
{
    return hc::precise_math::fmax(x, y);
}
__device__ double fmin(double x, double y)
{
    return hc::precise_math::fmin(x, y);
}
__device__ double fmod(double x, double y)
{
    return hc::precise_math::fmod(x, y);
}
__device__ double frexp(double x, int *y)
{
    return hc::precise_math::frexp(x, y);
}
__device__ double hypot(double x, double y)
{
    return hc::precise_math::hypot(x, y);
}
__device__ double ilogb(double x)
{
    return hc::precise_math::ilogb(x);
}
__device__ int isfinite(double x)
{
    return hc::precise_math::isfinite(x);
}
__device__ unsigned isinf(double x)
{
    return hc::precise_math::isinf(x);
}
__device__ unsigned isnan(double x)
{
    return hc::precise_math::isnan(x);
}
__device__ double j0(double x)
{
    return __hip_j0(x);
}
__device__ double j1(double x)
{
    return __hip_j1(x);
}
__device__ double jn(int n, double x)
{
    return __hip_jn(n, x);
}
__device__ double ldexp(double x, int exp)
{
    return hc::precise_math::ldexp(x, exp);
}
__device__ double lgamma(double x)
{
  double val = 0.0;
  double y = x - 1;
  while(y > 0){
    val += log(y--);
  }
  return val;
}
__device__ long long int llrint(double x)
{
    long long int y = hc::precise_math::round(x);
    return y;
}
__device__ long long int llround(double x)
{
    long long int y = hc::precise_math::round(x);
    return y;
}
__device__ double log(double x)
{
    return hc::precise_math::log(x);
}
__device__ double log10(double x)
{
    return hc::precise_math::log10(x);
}
__device__ double log1p(double x)
{
    return hc::precise_math::log1p(x);
}
__device__ double log2(double x)
{
    return hc::precise_math::log2(x);
}
__device__ double logb(double x)
{
    return hc::precise_math::logb(x);
}
__device__ long int lrint(double x)
{
    long int y = hc::precise_math::round(x);
    return y;
}
__device__ long int lround(double x)
{
    long int y = hc::precise_math::round(x);
    return y;
}
__device__ double modf(double x, double *iptr)
{
    return hc::precise_math::modf(x, iptr);
}
__device__ double nan(const char *tagp)
{
    return hc::precise_math::nan((int)*tagp);
}
__device__ double nearbyint(double x)
{
    return hc::precise_math::nearbyint(x);
}
__device__ double nextafter(double x, double y)
{
    return hc::precise_math::nextafter(x, y);
}
__device__ double norm(int x, const double *d)
{
  double val = 0;
  for(int i=0;i<x;i++){
    val += d[i]*d[i];
  }
  return hc::precise_math::sqrt(val);
}
__device__ double norm3d(double a, double b, double c)
{
    double x = a*a + b*b + c*c;
    return hc::precise_math::sqrt(x);
}
__device__ double norm4d(double a, double b, double c, double d)
{
    double x = a*a + b*b;
    double y = c*c + d*d;
    return hc::precise_math::sqrt(x+y);
}
__device__ double normcdf(double y)
{
     return ((hc::precise_math::erf(y)/HIP_SQRT_2) + 1)/2;
}
__device__ double normcdfinv(double y)
{
     return HIP_SQRT_2 * __hip_erfinv(2*y-1);
}
__device__ double pow(double x, double y)
{
    return hc::precise_math::pow(x, y);
}
__device__ double rcbrt(double x)
{
    return hc::precise_math::rcbrt(x);
}
__device__ double remainder(double x, double y)
{
    return hc::precise_math::remainder(x, y);
}
__device__ double remquo(double x, double y, int *quo)
{
    return hc::precise_math::remquo(x, y, quo);
}
__device__ double rhypot(double x, double y)
{
    return 1/hc::precise_math::sqrt(x*x + y*y);
}
__device__ double rint(double x)
{
    return hc::precise_math::round(x);
}
__device__ double rnorm3d(double a, double b, double c)
{
    return hc::precise_math::rsqrt(a*a + b*b + c*c);
}
__device__ double rnorm4d(double a, double b, double c, double d)
{
    return hc::precise_math::rsqrt(a*a + b*b + c*c + d*d);
}
__device__ double rnorm(int dim, const double* t)
{
    double x = 0.0;
    for(int i=0;i<dim;i++)
    {
        x = hc::precise_math::fma(t[i], t[i], x);
    }
    return 1/x;
}
__device__ double round(double x)
{
    return hc::precise_math::round(x);
}
__device__ double rsqrt(double x)
{
    return hc::precise_math::rsqrt(x);
}
__device__ double scalbln(double x, long int n)
{
    return hc::precise_math::scalb(x, n);
}
__device__ double scalbn(double x, int n)
{
    return hc::precise_math::scalbn(x, n);
}
__device__ int signbit(double x)
{
    return hc::precise_math::signbit(x);
}
__device__ double sin(double x)
{
    return hc::precise_math::sin(x);
}
__device__ void sincos(double x, double *sptr, double *cptr)
{
    *sptr = hc::precise_math::sin(x);
    *cptr = hc::precise_math::cos(x);
}
__device__ void sincospi(double x, double *sptr, double *cptr)
{
    *sptr = hc::precise_math::sinpi(x);
    *cptr = hc::precise_math::cospi(x);
}
__device__ double sinh(double x)
{
    return hc::precise_math::sinh(x);
}
__device__ double sinpi(double x)
{
    return hc::precise_math::sinpi(x);
}
__device__ double sqrt(double x)
{
    return hc::precise_math::sqrt(x);
}
__device__ double tan(double x)
{
    return hc::precise_math::tan(x);
}
__device__ double tanh(double x)
{
    return hc::precise_math::tanh(x);
}
__device__ double tgamma(double x)
{
    return hc::precise_math::tgamma(x);
}
__device__ double trunc(double x)
{
    return hc::precise_math::trunc(x);
}
__device__ double y0(double x)
{
    return __hip_y0(x);
}
__device__ double y1(double x)
{
    return __hip_y1(x);
}
__device__ double yn(int n, double x)
{
    return __hip_yn(n, x);
}


__host__ float cospif(float x)
{
    return std::cos(x*HIP_PI);
}

__host__ float fdividef(float x, float y)
{
  return x / y;
}

__host__ int isfinite(float x)
{
  return std::isfinite(x);
}

__host__ int signbit(float x)
{
  return std::signbit(x);
}

__host__ float sinpif(float x)
{
  return std::sin(x*HIP_PI);
}

__host__ float rsqrtf(float x)
{
  return 1 / std::sqrt(x);
}

__host__ float modff(float x, float *iptr)
{
    return std::modf(x, iptr);
}

__host__ double fdivide(double x, double y)
{
    return x/y;
}

__host__ float normcdff(float t)
{
     return (1 - std::erf(-t/std::sqrt(2)))/2;
}

__host__ double normcdf(double x)
{
     return (1 - std::erf(-x/std::sqrt(2)))/2;
}

__host__ float erfcxf(float x)
{
     return std::exp(x*x) * std::erfc(x);
}

__host__ double erfcx(double x)
{
     return std::exp(x*x) * std::erfc(x);
}

__host__ float rhypotf(float x, float y)
{
     return 1 / std::sqrt(x*x + y*y);
}

__host__ double rhypot(double x, double y)
{
    return 1 / std::sqrt(x*x + y*y);
}

__host__ float rcbrtf(float a)
{
    return 1 / std::cbrt(a);
}

__host__ double rcbrt(double a)
{
    return 1 / std::cbrt(a);
}

__host__ float normf(int dim, const float *a)
{
    float val = 0.0f;
    for(int i=0;i<dim;i++)
    {
        val = val + a[i] * a[i];
    }
    return val;
}

__host__ float rnormf(int dim, const float *t)
{
    float val = 0.0f;
    for(int i=0;i<dim;i++)
    {
        val = val + t[i] * t[i];
    }
    return 1 / std::sqrt(val);
}

__host__ double rnorm(int dim, const double *t)
{
    double val = 0.0;
    for(int i=0;i<dim;i++)
    {
        val = val + t[i] * t[i];
    }
    return 1 / std::sqrt(val);
}

__host__ float rnorm4df(float a, float b, float c, float d)
{
    return 1 / std::sqrt(a*a + b*b + c*c + d*d);
}

__host__ double rnorm4d(double a, double b, double c, double d)
{
    return 1 / std::sqrt(a*a + b*b + c*c + d*d);
}

__host__ float rnorm3df(float a, float b, float c)
{
    return 1 / std::sqrt(a*a + b*b + c*c);
}

__host__ double rnorm3d(double a, double b, double c)
{
    return 1 / std::sqrt(a*a + b*b + c*c);
}

__host__ void sincospif(float x, float *sptr, float *cptr)
{
    *sptr = std::sin(HIP_PI*x);
    *cptr = std::cos(HIP_PI*x);
}

__host__ void sincospi(double x, double *sptr, double *cptr)
{
    *sptr = std::sin(HIP_PI*x);
    *cptr = std::cos(HIP_PI*x);
}

__host__ float nextafterf(float x, float y)
{
    return std::nextafter(x, y);
}

__host__ double nextafter(double x, double y)
{
    return std::nextafter(x, y);
}

__host__ float norm3df(float a, float b, float c)
{
    return std::sqrt(a*a + b*b + c*c);
}

__host__ float norm4df(float a, float b, float c, float d)
{
    return std::sqrt(a*a + b*b + c*c + d*d);
}

__host__ double norm3d(double a, double b, double c)
{
    return std::sqrt(a*a + b*b + c*c);
}

__host__ double norm4d(double a, double b, double c, double d)
{
    return std::sqrt(a*a + b*b + c*c + d*d);
}

__host__ double sinpi(double a)
{
  return std::sin(HIP_PI * a);
}

__host__ double cospi(double a)
{
  return std::cos(HIP_PI * a);
}

__host__ int isfinite(double a)
{
  return std::isfinite(a);
}

__host__ double norm(int dim, const double *t)
{
  double val = 0;
  for(int i=0;i<dim;i++)
  {
    val += t[i]*t[i];
  }
  return std::sqrt(val);
}

__host__ double rsqrt(double x)
{
  return 1/std::sqrt(x);
}

__host__ int signbit(double x)
{
  return std::signbit(x);
}
