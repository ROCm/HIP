/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
__DEVICE__ inline uint64_t __make_mantissa_base8(const char* tagp);
__DEVICE__ inline uint64_t __make_mantissa_base10(const char* tagp);

__DEVICE__ inline uint64_t __make_mantissa_base16(const char* tagp);

__DEVICE__ inline uint64_t __make_mantissa(const char* tagp);
#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

// DOT FUNCTIONS
#if __HIP_CLANG_ONLY__
__DEVICE__ inline int amd_mixed_dot(short2 a, short2 b, int c, bool saturate);
__DEVICE__ inline uint amd_mixed_dot(ushort2 a, ushort2 b, uint c, bool saturate);
__DEVICE__ inline int amd_mixed_dot(char4 a, char4 b, int c, bool saturate);
__DEVICE__ inline uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate);
__DEVICE__ inline int amd_mixed_dot(int a, int b, int c, bool saturate);
__DEVICE__ inline uint amd_mixed_dot(uint a, uint b, uint c, bool saturate);
#endif

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
/// @defgroup MathFloat Single Precision Floating-point Mathematical Functions
/// @{
__DEVICE__ inline float abs(float x);
/// @brief Calculate the arc cosine of the input argument.
__DEVICE__ inline float acosf(float x);
/// @brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
__DEVICE__ inline float acoshf(float x);
/// @brief Calculate the arc sine of the input argument.
__DEVICE__ inline float asinf(float x);
/// @brief Calculate the arc hyperbolic sine of the input argument.
__DEVICE__ inline float asinhf(float x);
/// @brief Calculate the arc tangent of the ratio of first and second input arguments.
__DEVICE__ inline float atan2f(float x, float y);
/// @brief Calculate the arc tangent of the input argument.
__DEVICE__ inline float atanf(float x);
/// @brief Calculate the arc hyperbolic tangent of the input argument.
__DEVICE__ inline float atanhf(float x);
/// @brief Calculate the cube root of the input argument.
__DEVICE__ inline float cbrtf(float x);
/// @brief Calculate ceiling of the input argument.
__DEVICE__ inline float ceilf(float x);
/// @brief Create value with given magnitude, copying sign of second value.
__DEVICE__ inline float copysignf(float x, float y);
/// @brief Calculate the cosine of the input argument.
__DEVICE__ inline float cosf(float x);
/// @brief Calculate the hyperbolic cosine of the input argument.
__DEVICE__ inline float coshf(float x);
__DEVICE__ inline float cospif(float x);
__DEVICE__ inline float cyl_bessel_i0f(float x);
__DEVICE__ inline float cyl_bessel_i1f(float x);
/// @brief Calculate the complementary error function of the input argument.
__DEVICE__ inline float erfcf(float x);
/// @brief Calculate the inverse complementary function of the input argument.
__DEVICE__ inline float erfcinvf(float x);
/// @brief Calculate the scaled complementary error function of the input argument.
__DEVICE__ inline float erfcxf(float x);
/// @brief Calculate the error function of the input argument.
__DEVICE__ inline float erff(float x);
/// @brief Calculate the inverse error function of the input argument.
__DEVICE__ inline float erfinvf(float x);
/// @brief Calculate the base 10 exponential of the input argument.
__DEVICE__ inline float exp10f(float x);
/// @brief Calculate the base 2 exponential of the input argument.
__DEVICE__ inline float exp2f(float x);
/// @brief Calculate the base e exponential of the input argument.
__DEVICE__ inline float expf(float x);
/// @brief Calculate the base e exponential of the input argument, minus 1.
__DEVICE__ inline float expm1f(float x);
/// @brief Calculate the absolute value of its argument.
__DEVICE__ inline float fabsf(float x);
/// @brief Compute the positive difference between `x` and `y`.
__DEVICE__ inline float fdimf(float x, float y);
/// @brief Divide two floating point values.
__DEVICE__ inline float fdividef(float x, float y);
/// @brief Calculate the largest integer less than or equal to `x`.
__DEVICE__ inline float floorf(float x);
/// @brief Compute `x × y + z` as a single operation.
__DEVICE__ inline float fmaf(float x, float y, float z);
/// @brief Determine the maximum numeric value of the arguments.
__DEVICE__ inline float fmaxf(float x, float y);
/// @brief Determine the minimum numeric value of the arguments.
__DEVICE__ inline float fminf(float x, float y);
/// @brief Calculate the floating-point remainder of `x / y`.
__DEVICE__ inline float fmodf(float x, float y);
/// @brief Extract mantissa and exponent of a floating-point value.
__DEVICE__ inline float frexpf(float x, int* nptr);
/// @brief Calculate the square root of the sum of squares of two arguments.
__DEVICE__ inline float hypotf(float x, float y);
/// @brief Compute the unbiased integer exponent of the argument.
__DEVICE__ inline int ilogbf(float x);
/// @brief Determine whether argument is finite.
__DEVICE__ inline __RETURN_TYPE isfinite(float x);
/// @brief Determine whether argument is infinite.
__DEVICE__ inline __RETURN_TYPE isinf(float x);
/// @brief Determine whether argument is a NaN.
__DEVICE__ inline __RETURN_TYPE isnan(float x);
/// @brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
__DEVICE__ inline float j0f(float x);
/// @brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
__DEVICE__ inline float j1f(float x);
/// @brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
__DEVICE__ inline float jnf(int n, float x);
/// @brief Calculate the value of x ⋅ 2<sup>exp</sup>.
__DEVICE__ inline float ldexpf(float x, int e);
/// @brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
__DEVICE__ inline float lgammaf(float x);
/// @brief Round input to nearest integer value.
__DEVICE__ inline long long int llrintf(float x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long long int llroundf(float x);
/// @brief Calculate the base 10 logarithm of the input argument.
__DEVICE__ inline float log10f(float x);
/// @brief e( 1 + x ).
__DEVICE__ inline float log1pf(float x);
/// @brief Calculate the base 2 logarithm of the input argument.
__DEVICE__ inline float log2f(float x);
/// @brief Calculate the floating point representation of the exponent of the input argument.
__DEVICE__ inline float logbf(float x);
/// @brief Calculate the natural logarithm of the input argument.
__DEVICE__ inline float logf(float x);
/// @brief Round input to nearest integer value.
__DEVICE__ inline long int lrintf(float x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long int lroundf(float x);
/// @brief Break down the input argument into fractional and integral parts.
__DEVICE__ inline float modff(float x, float* iptr);
/// @brief Returns "Not a Number" value.
__DEVICE__ inline float nanf(const char* tagp);
/// @brief Round the input argument to the nearest integer.
__DEVICE__ inline float nearbyintf(float x);
/// @brief Returns next representable single-precision floating-point value after argument.
__DEVICE__ inline float nextafterf(float x, float y);
/// @brief Calculate the square root of the sum of squares of three coordinates of the argument.
__DEVICE__ inline float norm3df(float x, float y, float z);
/// @brief Calculate the square root of the sum of squares of four coordinates of the argument.
__DEVICE__ inline float norm4df(float x, float y, float z, float w);
/// @brief Calculate the standard normal cumulative distribution function.
__DEVICE__ inline float normcdff(float x);
/// @brief Calculate the inverse of the standard normal cumulative distribution function.
__DEVICE__ inline float normcdfinvf(float x);
/// @brief Calculate the square root of the sum of squares of any number of coordinates.
__DEVICE__ inline float normf(int dim, const float* a);
/// @brief Calculate the value of first argument to the power of second argument.
__DEVICE__ inline float powf(float x, float y);
__DEVICE__ inline float powif(float base, int iexp);
/// @brief Calculate the reciprocal cube root function.
__DEVICE__ inline float rcbrtf(float x);
/// @brief Compute single-precision floating-point remainder.
__DEVICE__ inline float remainderf(float x, float y);
/// @brief Compute single-precision floating-point remainder and part of quotient.
__DEVICE__ inline float remquof(float x, float y, int* quo);
/// @brief Calculate one over the square root of the sum of squares of two arguments.
__DEVICE__ inline float rhypotf(float x, float y);
/// @brief Round input to nearest integer value in floating-point.
__DEVICE__ inline float rintf(float x);
/// @brief Calculate one over the square root of the sum of squares of three coordinates of the argument.
__DEVICE__ inline float rnorm3df(float x, float y, float z);
/// @brief Calculate one over the square root of the sum of squares of four coordinates of the argument.
__DEVICE__ inline float rnorm4df(float x, float y, float z, float w);
/// @brief Calculate the reciprocal of square root of the sum of squares of any number of coordinates.
__DEVICE__ inline float rnormf(int dim, const float* a);
/// @brief Round to nearest integer value in floating-point.
__DEVICE__ inline float roundf(float x);
__DEVICE__ inline float rsqrtf(float x);
/// @brief Scale floating-point input by integer power of two.
__DEVICE__ inline float scalblnf(float x, long int n);
/// @brief Scale floating-point input by integer power of two.
__DEVICE__ inline float scalbnf(float x, int n);
/// @brief Return the sign bit of the input.
__DEVICE__ inline __RETURN_TYPE signbit(float x);
/// @brief Calculate the sine and cosine of the first input argument.
__DEVICE__ inline void sincosf(float x, float* sptr, float* cptr);
/// @brief Calculate the sine and cosine of the first input argument multiplied by PI.
__DEVICE__ inline void sincospif(float x, float* sptr, float* cptr);
/// @brief Calculate the sine of the input argument.
__DEVICE__ inline float sinf(float x);
/// @brief Calculate the hyperbolic sine of the input argument.
__DEVICE__ inline float sinhf(float x);
__DEVICE__ inline float sinpif(float x);
/// @brief Calculate the square root of the input argument.
__DEVICE__ inline float sqrtf(float x);
/// @brief Calculate the tangent of the input argument.
__DEVICE__ inline float tanf(float x);
/// @brief Calculate the hyperbolic tangent of the input argument.
__DEVICE__ inline float tanhf(float x);
/// @brief Calculate the gamma function of the input argument.
__DEVICE__ inline float tgammaf(float x);
/// @brief Truncate input argument to the integral part.
__DEVICE__ inline float truncf(float x);
/// @brief Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
__DEVICE__ inline float y0f(float x);
/// @brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
__DEVICE__ inline float y1f(float x);
/// @brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
__DEVICE__ inline float ynf(int n, float x);
/// @}
/// @defgroup MathFloatIntrinsics Single-precision Floating-point Intrinsics
/// @{
/// @brief Calculate the fast approximate cosine of the input argument.
__DEVICE__ inline float __cosf(float x);
__DEVICE__ inline float __exp10f(float x);
/// @brief Calculate the fast approximate base e exponential of the input argument.
__DEVICE__ inline float __expf(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fadd_rd(float x, float y);
#endif
__DEVICE__ inline float __fadd_rn(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fadd_ru(float x, float y);
__DEVICE__ inline float __fadd_rz(float x, float y);
__DEVICE__ inline float __fdiv_rd(float x, float y);
#endif
__DEVICE__ inline float __fdiv_rn(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fdiv_ru(float x, float y);
__DEVICE__ inline float __fdiv_rz(float x, float y);
#endif
__DEVICE__ inline float __fdividef(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fmaf_rd(float x, float y, float z);
#endif
__DEVICE__ inline float __fmaf_rn(float x, float y, float z);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fmaf_ru(float x, float y, float z);
__DEVICE__ inline float __fmaf_rz(float x, float y, float z);
__DEVICE__ inline float __fmul_rd(float x, float y);
#endif
__DEVICE__ inline float __fmul_rn(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fmul_ru(float x, float y) ;
__DEVICE__ inline float __fmul_rz(float x, float y);
__DEVICE__ inline float __frcp_rd(float x);
#endif
__DEVICE__ inline float __frcp_rn(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __frcp_ru(float x);
__DEVICE__ inline float __frcp_rz(float x);
#endif
/// @brief Compute `1 / √x` in round-to-nearest-even mode.
__DEVICE__ inline float __frsqrt_rn(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fsqrt_rd(float x);
#endif
/// @brief Compute `√x` in round-to-nearest-even mode.
__DEVICE__ inline float __fsqrt_rn(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fsqrt_ru(float x);
__DEVICE__ inline float __fsqrt_rz(float x);
__DEVICE__ inline float __fsub_rd(float x, float y);
#endif
__DEVICE__ inline float __fsub_rn(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fsub_ru(float x, float y);
__DEVICE__ inline float __fsub_rz(float x, float y);
#endif
/// @brief Calculate the fast approximate base 10 logarithm of the input argument.
__DEVICE__ inline float __log10f(float x);
/// @brief Calculate the fast approximate base 2 logarithm of the input argument.
__DEVICE__ inline float __log2f(float x);
/// @brief Calculate the fast approximate base e logarithm of the input argument.
__DEVICE__ inline float __logf(float x);
/// @brief Calculate the fast approximate of x<sup>y</sup>.
__DEVICE__ inline float __powf(float x, float y);
__DEVICE__ inline float __saturatef(float x);
__DEVICE__ inline void __sincosf(float x, float* sptr, float* cptr);
/// @brief Calculate the fast approximate sine of the input argument.
__DEVICE__ inline float __sinf(float x);
/// @brief Calculate the fast approximate tangent of the input argument.
__DEVICE__ inline float __tanf(float x);
/// @}

/// @defgroup MathDouble Double Precision Floating-point Mathematical Functions
/// @{
__DEVICE__ inline double abs(double x);
/// @brief Calculate the arc cosine of the input argument.
__DEVICE__ inline double acos(double x);
/// @brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
__DEVICE__ inline double acosh(double x);
/// @brief Calculate the arc sine of the input argument.
__DEVICE__ inline double asin(double x);
/// @brief  Calculate the arc hyperbolic sine of the input argument.
__DEVICE__ inline double asinh(double x);
/// @brief Calculate the arc tangent of the input argument.
__DEVICE__ inline double atan(double x);
/// @brief Calculate the arc tangent of the ratio of first and second input arguments.
__DEVICE__ inline double atan2(double x, double y);
/// @brief Calculate the arc hyperbolic tangent of the input argument.
__DEVICE__ inline double atanh(double x);
/// @brief Calculate the cube root of the input argument.
__DEVICE__ inline double cbrt(double x);
/// @brief Calculate ceiling of the input argument.
__DEVICE__ inline double ceil(double x);
/// @brief Create value with given magnitude, copying sign of second value.
__DEVICE__ inline double copysign(double x, double y);
/// @brief Calculate the cosine of the input argument.
__DEVICE__ inline double cos(double x) ;
/// @brief Calculate the hyperbolic cosine of the input argument.
__DEVICE__ inline double cosh(double x);
__DEVICE__ inline double cospi(double x);
__DEVICE__ inline double cyl_bessel_i0(double x);
__DEVICE__ inline double cyl_bessel_i1(double x);
/// @brief Calculate the error function of the input argument.
__DEVICE__ inline double erf(double x);
/// @brief Calculate the complementary error function of the input argument.
__DEVICE__ inline double erfc(double x);
/// @brief Calculate the inverse complementary function of the input argument.
__DEVICE__ inline double erfcinv(double x);
/// @brief Calculate the scaled complementary error function of the input argument.
__DEVICE__ inline double erfcx(double x);
/// @brief Calculate the inverse error function of the input argument.
__DEVICE__ inline double erfinv(double x);
/// @brief Calculate the base e exponential of the input argument.
__DEVICE__ inline double exp(double x);
/// @brief Calculate the base 10 exponential of the input argument.
__DEVICE__ inline double exp10(double x);
/// @brief Calculate the base 2 exponential of the input argument.
__DEVICE__ inline double exp2(double x);
/// @brief Calculate the base e exponential of the input argument, minus 1.
__DEVICE__ inline double expm1(double x);
/// @brief Calculate the absolute value of the input argument.
__DEVICE__ inline double fabs(double x);
/// @brief Compute the positive difference between `x` and `y`.
__DEVICE__ inline double fdim(double x, double y);
/// @brief Calculate the largest integer less than or equal to `x`.
__DEVICE__ inline double floor(double x);
/// @brief Compute `x × y + z` as a single operation.
__DEVICE__ inline double fma(double x, double y, double z);
/// @brief Determine the maximum numeric value of the arguments.
__DEVICE__ inline double fmax(double x, double y);
/// @brief Determine the minimum numeric value of the arguments.
__DEVICE__ inline double fmin(double x, double y);
/// @brief Calculate the floating-point remainder of `x / y`.
__DEVICE__ inline double fmod(double x, double y);
/// @brief Extract mantissa and exponent of a floating-point value.
__DEVICE__ inline double frexp(double x, int* nptr);
/// @brief Calculate the square root of the sum of squares of two arguments.
__DEVICE__ inline double hypot(double x, double y);
/// @brief Compute the unbiased integer exponent of the argument.
__DEVICE__ inline int ilogb(double x);
/// @brief Determine whether argument is finite.
__DEVICE__ inline __RETURN_TYPE isfinite(double x);
/// @brief Determine whether argument is infinite.
__DEVICE__ inline __RETURN_TYPE isinf(double x);
/// @brief Determine whether argument is a NaN.
__DEVICE__ inline __RETURN_TYPE isnan(double x);
/// @brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
__DEVICE__ inline double j0(double x);
/// @brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
__DEVICE__ inline double j1(double x);
/// @brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
__DEVICE__ inline double jn(int n, double x);
/// @brief Calculate the value of x ⋅ 2<sup>exp</sup>.
__DEVICE__ inline double ldexp(double x, int e);
/// @brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
__DEVICE__ inline double lgamma(double x);
/// @brief Round input to nearest integer value.
__DEVICE__ inline long long int llrint(double x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long long int llround(double x);
/// @brief Calculate the base e logarithm of the input argument.
__DEVICE__ inline double log(double x);
/// @brief Calculate the base 10 logarithm of the input argument.
__DEVICE__ inline double log10(double x);
/// @brief e( 1 + x ).
__DEVICE__ inline double log1p(double x);
/// @brief Calculate the base 2 logarithm of the input argument.
__DEVICE__ inline double log2(double x);
/// @brief Calculate the floating point representation of the exponent of the input argument.
__DEVICE__ inline double logb(double x);
/// @brief Round input to nearest integer value.
__DEVICE__ inline long int lrint(double x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long int lround(double x);
/// @brief Break down the input argument into fractional and integral parts.
__DEVICE__ inline double modf(double x, double* iptr);
/// @brief Returns "Not a Number" value.
__DEVICE__ inline double nan(const char* tagp);
/// @brief Round the input argument to the nearest integer.
__DEVICE__ inline double nearbyint(double x);
/// @brief Returns next representable single-precision floating-point value after argument.
__DEVICE__ inline double nextafter(double x, double y);
__DEVICE__ inline double norm(int dim, const double* a);
/// @brief Calculate the square root of the sum of squares of three coordinates of the argument.
__DEVICE__ inline double norm3d(double x, double y, double z);
/// @brief Calculate the square root of the sum of squares of four coordinates of the argument.
__DEVICE__ inline double norm4d(double x, double y, double z, double w);
/// @brief Calculate the standard normal cumulative distribution function.
__DEVICE__ inline double normcdf(double x);
/// @brief Calculate the inverse of the standard normal cumulative distribution function.
__DEVICE__ inline double normcdfinv(double x);
/// @brief Calculate the value of first argument to the power of second argument.
__DEVICE__ inline double pow(double x, double y);
__DEVICE__ inline double powi(double base, int iexp);
/// @brief Calculate the reciprocal cube root function.
__DEVICE__ inline double rcbrt(double x);
/// @brief Compute double-precision floating-point remainder.
__DEVICE__ inline double remainder(double x, double y);
/// @brief Compute double-precision floating-point remainder and part of quotient.
__DEVICE__ inline double remquo(double x, double y, int* quo);
/// @brief Calculate one over the square root of the sum of squares of two arguments.
__DEVICE__ inline double rhypot(double x, double y);
/// @brief Round input to nearest integer value in floating-point.
__DEVICE__ inline double rint(double x);
/// @brief Calculate the reciprocal of square root of the sum of squares of any number of coordinates.
__DEVICE__ inline double rnorm(int dim, const double* a);
/// @brief Calculate one over the square root of the sum of squares of three coordinates of the argument.
__DEVICE__ inline double rnorm3d(double x, double y, double z);
/// @brief Calculate one over the square root of the sum of squares of four coordinates of the argument.
__DEVICE__ inline double rnorm4d(double x, double y, double z, double w);
/// @brief Round to nearest integer value in floating-point.
__DEVICE__ inline double round(double x);
__DEVICE__ inline double rsqrt(double x);
/// @brief Scale floating-point input by integer power of two.
__DEVICE__ inline double scalbln(double x, long int n);
/// @brief Scale floating-point input by integer power of two.
__DEVICE__ inline double scalbn(double x, int n);
/// @brief Return the sign bit of the input.
__DEVICE__ inline __RETURN_TYPE signbit(double x);
/// @brief Calculate the sine of the input argument.
__DEVICE__ inline double sin(double x);
/// @brief Calculate the sine and cosine of the first input argument.
__DEVICE__ inline void sincos(double x, double* sptr, double* cptr);
/// @brief Calculate the sine and cosine of the first input argument multiplied by PI.
__DEVICE__ inline void sincospi(double x, double* sptr, double* cptr);
/// @brief Calculate the hyperbolic sine of the input argument.
__DEVICE__ inline double sinh(double x);
__DEVICE__ inline double sinpi(double x);
/// @brief Calculate the square root of the input argument.
__DEVICE__ inline double sqrt(double x);
/// @brief Calculate the tangent of the input argument.
__DEVICE__ inline double tan(double x);
/// @brief Calculate the hyperbolic tangent of the input argument.
__DEVICE__ inline double tanh(double x);
/// @brief Calculate the gamma function of the input argument.
__DEVICE__ inline double tgamma(double x);
/// @brief Truncate input argument to the integral part.
__DEVICE__ inline double trunc(double x);
__DEVICE__ inline double y0(double x);
/// @brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
__DEVICE__ inline double y1(double x);
/// @brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
__DEVICE__ inline double yn(int n, double x);

/// @}
/// @defgroup MathDoubleIntrinsics Double Precision Floating-point Intrinsics
/// @{
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __dadd_rd(double x, double y);
#endif
__DEVICE__ inline double __dadd_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __dadd_ru(double x, double y);
__DEVICE__ inline double __dadd_rz(double x, double y);
__DEVICE__ inline double __ddiv_rd(double x, double y);
#endif
__DEVICE__ inline double __ddiv_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __ddiv_ru(double x, double y);
__DEVICE__ inline double __ddiv_rz(double x, double y);
__DEVICE__ inline double __dmul_rd(double x, double y);
#endif
__DEVICE__ inline double __dmul_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __dmul_ru(double x, double y);
__DEVICE__ inline double __dmul_rz(double x, double y);
__DEVICE__ inline double __drcp_rd(double x);
#endif
__DEVICE__ inline double __drcp_rn(double x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __drcp_ru(double x);
__DEVICE__ inline double __drcp_rz(double x);
__DEVICE__ inline double __dsqrt_rd(double x);
#endif
/// @brief Compute `√x` in round-to-nearest-even mode.
__DEVICE__ inline double __dsqrt_rn(double x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __dsqrt_ru(double x);
__DEVICE__ inline double __dsqrt_rz(double x);
__DEVICE__ inline double __dsub_rd(double x, double y);
#endif
__DEVICE__ inline double __dsub_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __dsub_ru(double x, double y);
__DEVICE__ inline double __dsub_rz(double x, double y);
__DEVICE__ inline double __fma_rd(double x, double y, double z);
#endif
__DEVICE__ inline double __fma_rn(double x, double y, double z);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline double __fma_ru(double x, double y, double z);
__DEVICE__ inline double __fma_rz(double x, double y, double z);
#endif
/// @}

/// @defgroup MathInteger Integer Mathematical Functions
/// @{
__DEVICE__ inline int abs(int x);
__DEVICE__ inline long labs(long x);
__DEVICE__ inline long long llabs(long long x);

#if defined(__cplusplus)
    __DEVICE__ inline long abs(long x);
    __DEVICE__ inline long long abs(long long x);
#endif
/// @}

/// @brief Compute `x × y + z` as a single operation.
__DEVICE__ inline _Float16 fma(_Float16 x, _Float16 y, _Float16 z);
/// @brief Compute `x × y + z` as a single operation.
__DEVICE__ inline float fma(float x, float y, float z);

template<class T>
__DEVICE__ inline T min(T arg1, T arg2);

template<class T>
__DEVICE__ inline T max(T arg1, T arg2);

__DEVICE__ inline int min(int arg1, int arg2);
__DEVICE__ inline int max(int arg1, int arg2);
__DEVICE__ inline int min(uint32_t arg1, int arg2);
__DEVICE__ inline int max(uint32_t arg1, int arg2);
__DEVICE__ inline float max(float x, float y);
__DEVICE__ inline double max(double x, double y);
__DEVICE__ inline float min(float x, float y);
__DEVICE__ inline double min(double x, double y);

#if !defined(__HIPCC_RTC__)
__host__ inline static int min(int arg1, int arg2);
__host__ inline static int max(int arg1, int arg2);
#endif // !defined(__HIPCC_RTC__)

/// @brief Calculate the value of first argument to the power of second argument.
__DEVICE__ inline float pow(float base, int iexp);
/// @brief Calculate the value of first argument to the power of second argument.
__DEVICE__ inline double pow(double base, int iexp);
/// @brief Calculate the value of first argument to the power of second argument.
__DEVICE__ inline _Float16 pow(_Float16 base, int iexp);

#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__