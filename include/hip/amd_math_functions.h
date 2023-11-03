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


#ifndef HIP_INCLUDE_HIP_AMD_MATH_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_MATH_FUNCTIONS_H
/// @defgroup DeviceAPI Device APIs
/// @{
/// Defines the Device APIs. See the individual sections for more information.
/// @defgroup Math Math API
/// @{
/// Defines the Math API. See the individual sections for more information.
///

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
/** @addtogroup MathInteger
 * @{
 */
/// @brief Make base 8 (octal) mantissa from char array.
__DEVICE__ inline uint64_t __make_mantissa_base8(const char* tagp);
/// @brief Make base 10 (decimal) mantissa char array.
__DEVICE__ inline uint64_t __make_mantissa_base10(const char* tagp);
/// @brief Make base 16 (hexadecimal) mantissa char array.
__DEVICE__ inline uint64_t __make_mantissa_base16(const char* tagp);
/// @brief Make mantissa based on number format char array.
__DEVICE__ inline uint64_t __make_mantissa(const char* tagp);
/**
 * @}
 */
#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

// DOT FUNCTIONS
#if __HIP_CLANG_ONLY__
/** @defgroup MathIntegerIntrinsics Integer Intrinsics
 * @{
 */
/// Integer Intrinsics
/// @brief Returns \f$ \mathbf{a} \circ \mathbf{b} + c\f$ as a single operation with `a` and `b` as a 2-byte vector.
__DEVICE__ inline int amd_mixed_dot(short2 a, short2 b, int c, bool saturate);
/// @brief Returns \f$ \mathbf{a} \circ \mathbf{b} + c\f$ as a single operation with `a` and `b` as a 2-byte vector.
__DEVICE__ inline uint amd_mixed_dot(ushort2 a, ushort2 b, uint c, bool saturate);
/// @brief Returns \f$ \mathbf{a} \circ \mathbf{b} + c\f$ as a single operation with `a` and `b` as a 4-byte vector.
__DEVICE__ inline int amd_mixed_dot(char4 a, char4 b, int c, bool saturate);
/// @brief Returns \f$ \mathbf{a} \circ \mathbf{b} + c\f$ as a single operation with `a` and `b` as a 4-byte vector.
__DEVICE__ inline uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate);
/// @brief Returns \f$ \mathbf{a} \circ \mathbf{b} + c\f$ as a single operation with `a` and `b` as a 8-byte vector.
__DEVICE__ inline int amd_mixed_dot(int a, int b, int c, bool saturate);
/// @brief Returns \f$ \mathbf{a} \circ \mathbf{b} + c\f$ as a single operation with `a` and `b` as a 8-byte vector.
__DEVICE__ inline uint amd_mixed_dot(uint a, uint b, uint c, bool saturate);
/** @}
 */
#endif

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
/** @defgroup MathFloat Single Precision Floating-point Mathematical Functions
 * @{
 * Single Precision Floating-point Mathematical Functions
 */
/// @brief Returns the absolute value of `x`
__DEVICE__ inline float abs(float x);
/// @brief Returns the arc cosine of `x`.
__DEVICE__ inline float acosf(float x);
/// @brief Returns the nonnegative arc hyperbolic cosine of `x`.
__DEVICE__ inline float acoshf(float x);
/// @brief Returns the arc sine of `x`.
__DEVICE__ inline float asinf(float x);
/// @brief Returns the arc hyperbolic sine of `x`.
__DEVICE__ inline float asinhf(float x);
/// @brief Returns the arc tangent of the ratio of `x` and `y`.
__DEVICE__ inline float atan2f(float x, float y);
/// @brief Returns the arc tangent of `x`.
__DEVICE__ inline float atanf(float x);
/// @brief Returns the arc hyperbolic tangent of `x`.
__DEVICE__ inline float atanhf(float x);
/// @brief Returns the cube root of `x`.
__DEVICE__ inline float cbrtf(float x);
/// @brief Returns ceiling of `x`.
__DEVICE__ inline float ceilf(float x);
/// @brief Create value with given magnitude, copying sign of second value.
__DEVICE__ inline float copysignf(float x, float y);
/// @brief Returns the cosine of `x`.
__DEVICE__ inline float cosf(float x);
/// @brief Returns the hyperbolic cosine of `x`.
__DEVICE__ inline float coshf(float x);
/// @brief Returns the cosine of \f$ \pi x\f$.
__DEVICE__ inline float cospif(float x);
/// @brief Returns the value of the regular modified cylindrical Bessel function of order 0 for `x`.
__DEVICE__ inline float cyl_bessel_i0f(float x);
/// @brief Returns the value of the regular modified cylindrical Bessel function of order 1 for `x`.
__DEVICE__ inline float cyl_bessel_i1f(float x);
/// @brief Returns the complementary error function of `x`.
__DEVICE__ inline float erfcf(float x);
/// @brief Returns the inverse complementary function of `x`.
__DEVICE__ inline float erfcinvf(float x);
/// @brief Returns the scaled complementary error function of `x`.
__DEVICE__ inline float erfcxf(float x);
/// @brief Returns the error function of `x`.
__DEVICE__ inline float erff(float x);
/// @brief Returns the inverse error function of `x`.
__DEVICE__ inline float erfinvf(float x);
/// @brief Returns \f$ 10^x \f$.
__DEVICE__ inline float exp10f(float x);
/// @brief Returns \f$ 2^x \f$.
__DEVICE__ inline float exp2f(float x);
/// @brief Returns \f$ e^x \f$.
__DEVICE__ inline float expf(float x);
/// @brief Returns \f$ \ln x - 1 \f$
__DEVICE__ inline float expm1f(float x);
/// @brief Returns the absolute value of `x`
__DEVICE__ inline float fabsf(float x);
/// @brief Returns the positive difference between `x` and `y`.
__DEVICE__ inline float fdimf(float x, float y);
/// @brief Divide two floating point values.
__DEVICE__ inline float fdividef(float x, float y);
/// @brief Returns the largest integer less than or equal to `x`.
__DEVICE__ inline float floorf(float x);
/// @brief Returns \f$x \cdot y + z\f$ as a single operation.
__DEVICE__ inline float fmaf(float x, float y, float z);
/// @brief Determine the maximum numeric value of `x` and `y`.
__DEVICE__ inline float fmaxf(float x, float y);
/// @brief Determine the minimum numeric value of `x` and `y`.
__DEVICE__ inline float fminf(float x, float y);
/// @brief Returns the floating-point remainder of `x / y`.
__DEVICE__ inline float fmodf(float x, float y);
/// @brief Extract mantissa and exponent of `x`.
__DEVICE__ inline float frexpf(float x, int* nptr);
/// @brief Returns the square root of the sum of squares of `x` and `y`.
__DEVICE__ inline float hypotf(float x, float y);
/// @brief Returns the unbiased integer exponent of `x`.
__DEVICE__ inline int ilogbf(float x);
/// @brief Determine whether `x` is finite.
__DEVICE__ inline __RETURN_TYPE isfinite(float x);
/// @brief Determine whether `x` is infinite.
__DEVICE__ inline __RETURN_TYPE isinf(float x);
/// @brief Determine whether `x` is a NaN.
__DEVICE__ inline __RETURN_TYPE isnan(float x);
/// @brief Returns the value of the Bessel function of the first kind of order 0 for `x`.
__DEVICE__ inline float j0f(float x);
/// @brief Returns the value of the Bessel function of the first kind of order 1 for `x`.
__DEVICE__ inline float j1f(float x);
/// @brief Returns the value of the Bessel function of the first kind of order n for `x`.
__DEVICE__ inline float jnf(int n, float x);
/// @brief Returns the value of \f$x \cdot 2^{e}\f$ for `x` and `e`.
__DEVICE__ inline float ldexpf(float x, int e);
/// @brief Returns the natural logarithm of the absolute value of the gamma function of `x`.
__DEVICE__ inline float lgammaf(float x);
/// @brief Round `x` to nearest integer value.
__DEVICE__ inline long long int llrintf(float x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long long int llroundf(float x);
/// @brief Returns the base 10 logarithm of `x`.
__DEVICE__ inline float log10f(float x);
/// @brief Returns the natural logarithm of `x` + 1.
__DEVICE__ inline float log1pf(float x);
/// @brief Returns the base 2 logarithm of `x`.
__DEVICE__ inline float log2f(float x);
/// @brief Returns the floating point representation of the exponent of `x`.
__DEVICE__ inline float logbf(float x);
/// @brief Returns the natural logarithm of `x`.
__DEVICE__ inline float logf(float x);
/// @brief Round `x` to nearest integer value.
__DEVICE__ inline long int lrintf(float x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long int lroundf(float x);
/// @brief Break down `x` into fractional and integral parts.
__DEVICE__ inline float modff(float x, float* iptr);
/// @brief Returns "Not a Number" value.
__DEVICE__ inline float nanf(const char* tagp);
/// @brief Round `x` to the nearest integer.
__DEVICE__ inline float nearbyintf(float x);
/// @brief Returns next representable single-precision floating-point value after `x`.
__DEVICE__ inline float nextafterf(float x, float y);
/// @brief Returns the square root of the sum of squares of `x`, `y` and `z`.
__DEVICE__ inline float norm3df(float x, float y, float z);
/// @brief Returns the square root of the sum of squares of `x`, `y`, `z` and `w`.
__DEVICE__ inline float norm4df(float x, float y, float z, float w);
/// @brief Returns the standard normal cumulative distribution function.
__DEVICE__ inline float normcdff(float x);
/// @brief Returns the inverse of the standard normal cumulative distribution function.
__DEVICE__ inline float normcdfinvf(float x);
/// @brief Returns the square root of the sum of squares of any number of coordinates.
__DEVICE__ inline float normf(int dim, const float* a);
/// @brief Returns \f$ x^y \f$.
__DEVICE__ inline float powf(float x, float y);
/// @brief Returns the value of first argument to the power of second argument.
__DEVICE__ inline float powif(float base, int iexp);
/// @brief Returns the reciprocal cube root function.
__DEVICE__ inline float rcbrtf(float x);
/// @brief Returns single-precision floating-point remainder.
__DEVICE__ inline float remainderf(float x, float y);
/// @brief Returns single-precision floating-point remainder and part of quotient.
__DEVICE__ inline float remquof(float x, float y, int* quo);
/// @brief Returns one over the square root of the sum of squares of `x` and `y`.
__DEVICE__ inline float rhypotf(float x, float y);
/// @brief Round `x` to nearest integer value in floating-point.
__DEVICE__ inline float rintf(float x);
/// @brief Returns one over the square root of the sum of squares of `x`, `y` and `z`.
__DEVICE__ inline float rnorm3df(float x, float y, float z);
/// @brief Returns one over the square root of the sum of squares of `x`, `y`, `z` and `w`.
__DEVICE__ inline float rnorm4df(float x, float y, float z, float w);
/// @brief Returns the reciprocal of square root of the sum of squares of any number of coordinates.
__DEVICE__ inline float rnormf(int dim, const float* a);
/// @brief Round to nearest integer value in floating-point.
__DEVICE__ inline float roundf(float x);
/// @brief Returns the reciprocal of the square root of `x`.
__DEVICE__ inline float rsqrtf(float x);
/// @brief Scale `x` by \f$ 2^n \f$.
__DEVICE__ inline float scalblnf(float x, long int n);
/// @brief Scale `x` by \f$ 2^n \f$.
__DEVICE__ inline float scalbnf(float x, int n);
/// @brief Return the sign bit of `x`.
__DEVICE__ inline __RETURN_TYPE signbit(float x);
/// @brief Returns the sine and cosine of `x`.
__DEVICE__ inline void sincosf(float x, float* sptr, float* cptr);
/// @brief Returns the sine and cosine of \f$ \pi x\f$.
__DEVICE__ inline void sincospif(float x, float* sptr, float* cptr);
/// @brief Returns the sine of `x`.
__DEVICE__ inline float sinf(float x);
/// @brief Returns the hyperbolic sine of `x`.
__DEVICE__ inline float sinhf(float x);
/// @brief Returns the hyperbolic sine of \f$ \pi x\f$.
__DEVICE__ inline float sinpif(float x);
/// @brief Returns the square root of `x`.
__DEVICE__ inline float sqrtf(float x);
/// @brief Returns the tangent of `x`.
__DEVICE__ inline float tanf(float x);
/// @brief Returns the hyperbolic tangent of `x`.
__DEVICE__ inline float tanhf(float x);
/// @brief Returns the gamma function of `x`.
__DEVICE__ inline float tgammaf(float x);
/// @brief Truncate `x` to the integral part.
__DEVICE__ inline float truncf(float x);
/// @brief Returns the value of the Bessel function of the second kind of order 0 for `x`.
__DEVICE__ inline float y0f(float x);
/// @brief Returns the value of the Bessel function of the second kind of order 1 for `x`.
__DEVICE__ inline float y1f(float x);
/// @brief Returns the value of the Bessel function of the second kind of order n for `x`.
__DEVICE__ inline float ynf(int n, float x);
/// @}
/** @defgroup MathFloatIntrinsics Single Precision Floating-point Intrinsics
 * @{
 * Single Precision Floating-point Intrinsics
 */
/// @brief Returns the fast approximate cosine of `x`.
__DEVICE__ inline float __cosf(float x);
/// @brief Returns the fast approximate for \f$ 10^x \f$.
__DEVICE__ inline float __exp10f(float x);
/// @brief Returns the fast approximate for \f$ e^x \f$.
__DEVICE__ inline float __expf(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Add two floating-point values in round-down mode.
__DEVICE__ inline float __fadd_rd(float x, float y);
#endif
/// @brief Add two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline float __fadd_rn(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Add two floating-point values in round-up mode.
__DEVICE__ inline float __fadd_ru(float x, float y);
/// @brief Add two floating-point values in round-towards-zero mode.
__DEVICE__ inline float __fadd_rz(float x, float y);
/// @brief Divide two floating-point values in round-down mode.
__DEVICE__ inline float __fdiv_rd(float x, float y);
#endif
/// @brief Divide two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline float __fdiv_rn(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Divide two floating-point values in round-up mode.
__DEVICE__ inline float __fdiv_ru(float x, float y);
/// @brief Divide two floating-point values in round-towards-zero mode.
__DEVICE__ inline float __fdiv_rz(float x, float y);
#endif
/// @brief Returns the fast approximate of `x / y`.
__DEVICE__ inline float __fdividef(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Returns \f$x \cdot y + z\f$ as a single operation, in round-down mode.
__DEVICE__ inline float __fmaf_rd(float x, float y, float z);
#endif
/// @brief Returns \f$x \cdot y + z\f$ as a single operation, in round-to-nearest-even mode.
__DEVICE__ inline float __fmaf_rn(float x, float y, float z);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Returns \f$x \cdot y + z\f$ as a single operation, in round-up mode.
__DEVICE__ inline float __fmaf_ru(float x, float y, float z);
/// @brief Returns \f$x \cdot y + z\f$ as a single operation, in round-towards-zero mode.
__DEVICE__ inline float __fmaf_rz(float x, float y, float z);
/// @brief Multiply two floating-point values in round-down mode.
__DEVICE__ inline float __fmul_rd(float x, float y);
#endif
/// @brief Multiply two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline float __fmul_rn(float x, float y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Multiply two floating-point values in round-up mode.
__DEVICE__ inline float __fmul_ru(float x, float y);
/// @brief Multiply two floating-point values in round-towards-zero mode.
__DEVICE__ inline float __fmul_rz(float x, float y);
/// @brief Returns 1 / x in round-down mode.
__DEVICE__ inline float __frcp_rd(float x);
#endif
/// @brief Returns 1 / x in round-to-nearest-even mode.
__DEVICE__ inline float __frcp_rn(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Returns 1 / x in round-up mode.
__DEVICE__ inline float __frcp_ru(float x);
/// @brief Returns 1 / x in round-towards-zero mode.
__DEVICE__ inline float __frcp_rz(float x);
#endif
/// @brief Returns \f$ 1 / \sqrt{x}\f$ in round-to-nearest-even mode.
__DEVICE__ inline float __frsqrt_rn(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fsqrt_rd(float x);
#endif
/// @brief Returns \f$\sqrt{x}\f$ in round-to-nearest-even mode.
__DEVICE__ inline float __fsqrt_rn(float x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Returns \f$\sqrt{x}\f$ in round-up mode.
__DEVICE__ inline float __fsqrt_ru(float x);
/// @brief Returns \f$\sqrt{x}\f$ in round-towards-zero mode.
__DEVICE__ inline float __fsqrt_rz(float x);
/// @brief Subtract two floating-point values in round-down mode.
__DEVICE__ inline float __fsub_rd(float x, float y);
#endif
/// @brief Subtract two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline float __fsub_rn(float x, float y);
/// @brief Subtract two floating-point values in round-up mode.
#if defined OCML_BASIC_ROUNDED_OPERATIONS
__DEVICE__ inline float __fsub_ru(float x, float y);
/// @brief Subtract two floating-point values in round-towards-zero mode.
__DEVICE__ inline float __fsub_rz(float x, float y);
#endif
/// @brief Returns the fast approximate for base 10 logarithm of `x`.
__DEVICE__ inline float __log10f(float x);
/// @brief Returns the fast approximate for base 2 logarithm of `x`.
__DEVICE__ inline float __log2f(float x);
/// @brief Returns the fast approximate for natural logarithm of `x`.
__DEVICE__ inline float __logf(float x);
/// @brief Returns the fast approximate of \f$ x^y \f$
__DEVICE__ inline float __powf(float x, float y);
/// @brief Clamp `x` to [+0.0, 1.0].
__DEVICE__ inline float __saturatef(float x);
/// @brief Returns the fast approximate of sine and cosine of `x`.
__DEVICE__ inline void __sincosf(float x, float* sptr, float* cptr);
/// @brief Returns the fast approximate sine of `x`.
__DEVICE__ inline float __sinf(float x);
/// @brief Returns the fast approximate tangent of `x`.
__DEVICE__ inline float __tanf(float x);
/** @}
 *
 * @defgroup MathDouble Double Precision Floating-point Mathematical Functions
 * @{
 * Double Precision Floating-point Mathematical Functions
 */
/// @brief Returns the absolute value of `x`
__DEVICE__ inline double abs(double x);
/// @brief Returns the arc cosine of `x`.
__DEVICE__ inline double acos(double x);
/// @brief Returns the nonnegative arc hyperbolic cosine of `x`.
__DEVICE__ inline double acosh(double x);
/// @brief Returns the arc sine of `x`.
__DEVICE__ inline double asin(double x);
/// @brief Returns the arc hyperbolic sine of `x`.
__DEVICE__ inline double asinh(double x);
/// @brief Returns the arc tangent of `x`.
__DEVICE__ inline double atan(double x);
/// @brief Returns the arc tangent of the ratio of `x` and `y`.
__DEVICE__ inline double atan2(double x, double y);
/// @brief Returns the arc hyperbolic tangent of `x`.
__DEVICE__ inline double atanh(double x);
/// @brief Returns the cube root of `x`.
__DEVICE__ inline double cbrt(double x);
/// @brief Returns ceiling of `x`.
__DEVICE__ inline double ceil(double x);
/// @brief Create value with given magnitude, copying sign of second value.
__DEVICE__ inline double copysign(double x, double y);
/// @brief Returns the cosine of `x`.
__DEVICE__ inline double cos(double x) ;
/// @brief Returns the hyperbolic cosine of `x`.
__DEVICE__ inline double cosh(double x);
/// @brief Returns the cosine of \f$ x\pi\f$.
__DEVICE__ inline double cospi(double x);
/// @brief Returns the value of the regular modified cylindrical Bessel function of order 0 for `x`.
__DEVICE__ inline double cyl_bessel_i0(double x);
/// @brief Returns the value of the regular modified cylindrical Bessel function of order 1 for `x`.
__DEVICE__ inline double cyl_bessel_i1(double x);
/// @brief Returns the error function of `x`.
__DEVICE__ inline double erf(double x);
/// @brief Returns the complementary error function of `x`.
__DEVICE__ inline double erfc(double x);
/// @brief Returns the inverse complementary function of `x`.
__DEVICE__ inline double erfcinv(double x);
/// @brief Returns the scaled complementary error function of `x`.
__DEVICE__ inline double erfcx(double x);
/// @brief Returns the inverse error function of `x`.
__DEVICE__ inline double erfinv(double x);
/// @brief Returns \f$ e^x \f$.
__DEVICE__ inline double exp(double x);
/// @brief Returns \f$ 10^x \f$.
__DEVICE__ inline double exp10(double x);
/// @brief Returns \f$ 2^x \f$.
__DEVICE__ inline double exp2(double x);
/// @brief Returns \f$ e^x -1\f$ for `x`.
__DEVICE__ inline double expm1(double x);
/// @brief Returns the absolute value of `x`.
__DEVICE__ inline double fabs(double x);
/// @brief Returns the positive difference between `x` and `y`.
__DEVICE__ inline double fdim(double x, double y);
/// @brief Returns the largest integer less than or equal to `x`.
__DEVICE__ inline double floor(double x);
/// @brief Returns \f$x \cdot y + z\f$ as a single operation.
__DEVICE__ inline double fma(double x, double y, double z);
/// @brief Determine the maximum numeric value of `x` and `y`.
__DEVICE__ inline double fmax(double x, double y);
/// @brief Determine the minimum numeric value of `x` and `y`.
__DEVICE__ inline double fmin(double x, double y);
/// @brief Returns the floating-point remainder of `x / y`.
__DEVICE__ inline double fmod(double x, double y);
/// @brief Extract mantissa and exponent of `x`.
__DEVICE__ inline double frexp(double x, int* nptr);
/// @brief Returns the square root of the sum of squares of `x` and `y`.
__DEVICE__ inline double hypot(double x, double y);
/// @brief Returns the unbiased integer exponent of `x`.
__DEVICE__ inline int ilogb(double x);
/// @brief Determine whether `x` is finite.
__DEVICE__ inline __RETURN_TYPE isfinite(double x);
/// @brief Determine whether `x` is infinite.
__DEVICE__ inline __RETURN_TYPE isinf(double x);
/// @brief Determine whether `x` is a NaN.
__DEVICE__ inline __RETURN_TYPE isnan(double x);
/// @brief Returns the value of the Bessel function of the first kind of order 0 for `x`.
__DEVICE__ inline double j0(double x);
/// @brief Returns the value of the Bessel function of the first kind of order 1 for `x`.
__DEVICE__ inline double j1(double x);
/// @brief Returns the value of the Bessel function of the first kind of order n for `x`.
__DEVICE__ inline double jn(int n, double x);
/// @brief Returns the value of \f$x \cdot 2^{e}\f$ for `x` and `e`.
__DEVICE__ inline double ldexp(double x, int e);
/// @brief Returns the natural logarithm of the absolute value of the gamma function of `x`.
__DEVICE__ inline double lgamma(double x);
/// @brief Round `x` to nearest integer value.
__DEVICE__ inline long long int llrint(double x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long long int llround(double x);
/// @brief Returns the natural logarithm of `x`.
__DEVICE__ inline double log(double x);
/// @brief Returns the base 10 logarithm of `x`.
__DEVICE__ inline double log10(double x);
/// @brief Returns the natural logarithm of `x` + 1.
__DEVICE__ inline double log1p(double x);
/// @brief Returns the base 2 logarithm of `x`.
__DEVICE__ inline double log2(double x);
/// @brief Returns the floating point representation of the exponent of `x`.
__DEVICE__ inline double logb(double x);
/// @brief Round `x` to nearest integer value.
__DEVICE__ inline long int lrint(double x);
/// @brief Round to nearest integer value.
__DEVICE__ inline long int lround(double x);
/// @brief Break down `x` into fractional and integral parts.
__DEVICE__ inline double modf(double x, double* iptr);
/// @brief Returns "Not a Number" value.
__DEVICE__ inline double nan(const char* tagp);
/// @brief Round `x` to the nearest integer.
__DEVICE__ inline double nearbyint(double x);
/// @brief Returns next representable single-precision floating-point value after `x`.
__DEVICE__ inline double nextafter(double x, double y);
/// @brief Returns the square root of the sum of squares of any number of coordinates.
__DEVICE__ inline double norm(int dim, const double* a);
/// @brief Returns the square root of the sum of squares of `x`, `y` and `z`.
__DEVICE__ inline double norm3d(double x, double y, double z);
/// @brief Returns the square root of the sum of squares of `x`, `y`, `z` and `w`.
__DEVICE__ inline double norm4d(double x, double y, double z, double w);
/// @brief Returns the standard normal cumulative distribution function.
__DEVICE__ inline double normcdf(double x);
/// @brief Returns the inverse of the standard normal cumulative distribution function.
__DEVICE__ inline double normcdfinv(double x);
/// @brief Returns \f$ x^y \f$.
__DEVICE__ inline double pow(double x, double y);
/// @brief Returns the value of first argument to the power of second argument.
__DEVICE__ inline double powi(double base, int iexp);
/// @brief Returns the reciprocal cube root function.
__DEVICE__ inline double rcbrt(double x);
/// @brief Returns double-precision floating-point remainder.
__DEVICE__ inline double remainder(double x, double y);
/// @brief Returns double-precision floating-point remainder and part of quotient.
__DEVICE__ inline double remquo(double x, double y, int* quo);
/// @brief Returns one over the square root of the sum of squares of `x` and `y`.
__DEVICE__ inline double rhypot(double x, double y);
/// @brief Round `x` to nearest integer value in floating-point.
__DEVICE__ inline double rint(double x);
/// @brief Returns the reciprocal of square root of the sum of squares of any number of coordinates.
__DEVICE__ inline double rnorm(int dim, const double* a);
/// @brief Returns one over the square root of the sum of squares of `x`, `y` and `z`.
__DEVICE__ inline double rnorm3d(double x, double y, double z);
/// @brief Returns one over the square root of the sum of squares of `x`, `y`, `z` and `w`.
__DEVICE__ inline double rnorm4d(double x, double y, double z, double w);
/// @brief Round to nearest integer value in floating-point.
__DEVICE__ inline double round(double x);
/// @brief Returns the reciprocal of the square root of `x`.
__DEVICE__ inline double rsqrt(double x);
/// @brief Scale `x` by \f$ 2^n \f$.
__DEVICE__ inline double scalbln(double x, long int n);
/// @brief Scale `x` by \f$ 2^n \f$.
__DEVICE__ inline double scalbn(double x, int n);
/// @brief Return the sign bit of `x`.
__DEVICE__ inline __RETURN_TYPE signbit(double x);
/// @brief Returns the sine of `x`.
__DEVICE__ inline double sin(double x);
/// @brief Returns the sine and cosine of `x`.
__DEVICE__ inline void sincos(double x, double* sptr, double* cptr);
/// @brief Returns the sine and cosine of \f$ \pi x\f$.
__DEVICE__ inline void sincospi(double x, double* sptr, double* cptr);
/// @brief Returns the hyperbolic sine of `x`.
__DEVICE__ inline double sinh(double x);
/// @brief Returns the hyperbolic sine of \f$ \pi x\f$.
__DEVICE__ inline double sinpi(double x);
/// @brief Returns the square root of `x`.
__DEVICE__ inline double sqrt(double x);
/// @brief Returns the tangent of `x`.
__DEVICE__ inline double tan(double x);
/// @brief Returns the hyperbolic tangent of `x`.
__DEVICE__ inline double tanh(double x);
/// @brief Returns the gamma function of `x`.
__DEVICE__ inline double tgamma(double x);
/// @brief Truncate `x` to the integral part.
__DEVICE__ inline double trunc(double x);
/// @brief Returns the value of the Bessel function of the second kind of order 0 for `x`.
__DEVICE__ inline double y0(double x);
/// @brief Returns the value of the Bessel function of the second kind of order 1 for `x`.
__DEVICE__ inline double y1(double x);
/// @brief Returns the value of the Bessel function of the second kind of order n for `x`.
__DEVICE__ inline double yn(int n, double x);
/** @}
 *
 * @defgroup MathDoubleIntrinsics Double Precision Floating-point Intrinsics
 * @{
 */
/// Double Precision Floating-point Intrinsics
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Add two floating-point values in round-down mode.
__DEVICE__ inline double __dadd_rd(double x, double y);
#endif
/// @brief Add two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline double __dadd_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Add two floating-point values in round-up mode.
__DEVICE__ inline double __dadd_ru(double x, double y);
/// @brief Add two floating-point values in round-towards-zero mode.
__DEVICE__ inline double __dadd_rz(double x, double y);
/// @brief Divide two floating-point values in round-down mode.
__DEVICE__ inline double __ddiv_rd(double x, double y);
#endif
/// @brief Divide two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline double __ddiv_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Divide two floating-point values in round-up mode.
__DEVICE__ inline double __ddiv_ru(double x, double y);
/// @brief Divide two floating-point values in round-towards-zero mode.
__DEVICE__ inline double __ddiv_rz(double x, double y);
/// @brief Multiply two floating-point values in round-down mode.
__DEVICE__ inline double __dmul_rd(double x, double y);
#endif
/// @brief Multiply two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline double __dmul_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Multiply two floating-point values in round-up mode.
__DEVICE__ inline double __dmul_ru(double x, double y);
/// @brief Multiply two floating-point values in round-towards-zero mode.
__DEVICE__ inline double __dmul_rz(double x, double y);
/// @brief Returns 1 / x in round-down mode.
__DEVICE__ inline double __drcp_rd(double x);
#endif
/// @brief Returns 1 / x in round-to-nearest-even mode.
__DEVICE__ inline double __drcp_rn(double x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Returns 1 / x in round-up mode.
__DEVICE__ inline double __drcp_ru(double x);
/// @brief Returns 1 / x in round-towards-zero mode.
__DEVICE__ inline double __drcp_rz(double x);
/// @brief Returns \f$\sqrt{x}\f$ in round-down mode.
__DEVICE__ inline double __dsqrt_rd(double x);
#endif
/// @brief Returns \f$\sqrt{x}\f$ in round-to-nearest-even mode.
__DEVICE__ inline double __dsqrt_rn(double x);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Returns \f$\sqrt{x}\f$ in round-up mode.
__DEVICE__ inline double __dsqrt_ru(double x);
/// @brief Returns \f$\sqrt{x}\f$ in round-towards-zero mode.
__DEVICE__ inline double __dsqrt_rz(double x);
/// @brief Subtract two floating-point values in round-down mode.
__DEVICE__ inline double __dsub_rd(double x, double y);
#endif
/// @brief Subtract two floating-point values in round-to-nearest-even mode.
__DEVICE__ inline double __dsub_rn(double x, double y);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Subtract two floating-point values in round-up mode.
__DEVICE__ inline double __dsub_ru(double x, double y);
/// @brief Subtract two floating-point values in round-towards-zero mode.
__DEVICE__ inline double __dsub_rz(double x, double y);
/// @brief Returns \f$x \cdot y + z\f$ as a single operation in round-down mode.
__DEVICE__ inline double __fma_rd(double x, double y, double z);
#endif
/// @brief Returns \f$x \cdot y + z\f$ as a single operation in round-to-nearest-even mode.
__DEVICE__ inline double __fma_rn(double x, double y, double z);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
/// @brief Returns \f$x \cdot y + z\f$ as a single operation in round-up mode.
__DEVICE__ inline double __fma_ru(double x, double y, double z);
/// @brief Returns \f$x \cdot y + z\f$ as a single operation in round-towards-zero mode.
__DEVICE__ inline double __fma_rz(double x, double y, double z);
#endif
/** @}
 *
 * @defgroup MathInteger Integer Mathematical Functions
 * @{
 * Integer Mathematical Functions
 */

/// @brief Returns the absolute value of `x`.
__DEVICE__ inline int abs(int x);
/// @brief Returns the absolute value of `x`.
__DEVICE__ inline long labs(long x);
/// @brief Returns the absolute value of `x`.
__DEVICE__ inline long long llabs(long long x);

#if defined(__cplusplus)
/// @brief Returns the absolute value of `x` (override for C++).
__DEVICE__ inline long abs(long x);
/// @brief Returns the absolute value of `x` (override for C++).
__DEVICE__ inline long long abs(long long x);
#endif
/** @}
 *
 * @defgroup MathFloat16 _Float16 Floating-point Functions

 * @{
 * _Float16 Precision Mathematical Functions
 */
/// @brief Returns \f$x \cdot y + z\f$ as a single operation.
__DEVICE__ inline _Float16 fma(_Float16 x, _Float16 y, _Float16 z);


/** @} */
/// \ingroup MathFloat
/// @brief Returns \f$x \cdot y + z\f$ as a single operation.
__DEVICE__ inline float fma(float x, float y, float z);

/**
 *
 * @defgroup MathTemplate Template Functions

 * @{
 * `_Float16` Precision Mathematical Functions
 */
/// @brief Returns the minimum value of the input arguments.
template<class T>
__DEVICE__ inline T min(T arg1, T arg2);

/// @brief Returns the maximum value of the input arguments.
template<class T>
__DEVICE__ inline T max(T arg1, T arg2);
/** @}
 *
 * \addtogroup MathInteger
 * @{
 */
/// @brief Returns the minimum value of the input `int` arguments.
__DEVICE__ inline int min(int arg1, int arg2);
/// \ingroup MathInteger
/// @brief Returns the maximum value of the input `int` arguments.
__DEVICE__ inline int max(int arg1, int arg2);
/// @brief Returns the minimum value of the input `uint_32` arguments.
__DEVICE__ inline int min(uint32_t arg1, int arg2);
/// @brief Returns the maximum value of the input `uint_32` arguments.
__DEVICE__ inline int max(uint32_t arg1, int arg2);
/**
 * @}
 */
/// \ingroup MathFloat
/// @brief Returns the minimum value of the input `float` arguments.
__DEVICE__ inline float max(float x, float y);
/// \ingroup MathDouble
/// @brief Returns the maximum value of the input `double` arguments.
__DEVICE__ inline double max(double x, double y);
/// \ingroup MathFloat
/// @brief Returns the minimum value of the input `float` arguments.
__DEVICE__ inline float min(float x, float y);
/// \ingroup MathDouble
/// @brief Returns the maximum value of the input `double` arguments.
__DEVICE__ inline double min(double x, double y);

/** \addtogroup MathInteger
 * @{
 */
#if !defined(__HIPCC_RTC__)
/// @brief Returns the minimum value of the input `int` arguments on host.
__host__ inline static int min(int arg1, int arg2);
/// @brief Returns the maximum value of the input `int` arguments on host.
__host__ inline static int max(int arg1, int arg2);
#endif // !defined(__HIPCC_RTC__)
/**
 * @}
 */

/// \ingroup MathFloat
/// @brief Returns the value of first argument to the power of second argument.
__DEVICE__ inline float pow(float base, int iexp);
/// \ingroup MathDouble
/// @brief Returns the value of first argument to the power of second argument.
__DEVICE__ inline double pow(double base, int iexp);
/// \ingroup MathFloat16
/// @brief Returns the value of first argument to the power of second argument.
__DEVICE__ inline _Float16 pow(_Float16 base, int iexp);

#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

// doxygen end Math API
/// @}
// doxygen end Device APIs
/// @}

#endif // HIP_INCLUDE_HIP_AMD_MATH_FUNCTIONS_H