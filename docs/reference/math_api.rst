.. meta::
  :description: This chapter describes the built-in math functions that are accessible in HIP. 
  :keywords: AMD, ROCm, HIP, CUDA, math functions, HIP math functions

.. _math_api_reference:

********************************************************************************
HIP math API
********************************************************************************

HIP-Clang supports a set of math operations that are callable from the device. HIP supports most of the device functions supported by NVIDIA CUDA. These are described in the following sections.

Single precision mathematical functions
=======================================


Following is the list of supported single precision mathematical functions.

.. list-table:: Single precision mathematical functions

    * - **Function**
      - **Supported on Host**
      - **Supported on Device**

    * - | ``float abs(float x)``
        | Returns the absolute value of :math:`x`
      - ✓
      - ✓

    * - | ``float acosf(float x)``
        | Returns the arc cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float acoshf(float x)``
        | Returns the nonnegative arc hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float asinf(float x)``
        | Returns the arc sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float asinhf(float x)``
        | Returns the arc hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float atanf(float x)``
        | Returns the arc tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float atan2f(float x, float y)``
        | Returns the arc tangent of the ratio of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float atanhf(float x)``
        | Returns the arc hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float cbrtf(float x)``
        | Returns the cube root of :math:`x`.
      - ✓
      - ✓

    * - | ``float ceilf(float x)``
        | Returns ceiling of :math:`x`.
      - ✓
      - ✓

    * - | ``float copysignf(float x, float y)``
        | Create value with given magnitude, copying sign of second value.
      - ✓
      - ✓

    * - | ``float cosf(float x)``
        | Returns the cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float coshf(float x)``
        | Returns the hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``float cospif(float x)``
        | Returns the cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``float cyl_bessel_i0f(float x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 0 for :math:`x`.
      - ✗
      - ✗

    * - | ``float cyl_bessel_i1f(float x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 1 for :math:`x`.
      - ✗
      - ✗

    * - | ``float erff(float x)``
        | Returns the error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfcf(float x)``
        | Returns the complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfcinvf(float x)``
        | Returns the inverse complementary function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfcxf(float x)``
        | Returns the scaled complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float erfinvf(float x)``
        | Returns the inverse error function of :math:`x`.
      - ✓
      - ✓

    * - | ``float expf(float x)``
        | Returns :math:`e^x`.
      - ✓
      - ✓

    * - | ``float exp10f(float x)``
        | Returns :math:`10^x`.
      - ✓
      - ✓

    * - | ``float exp2f( float x)``
        | Returns :math:`2^x`.
      - ✓
      - ✓

    * - | ``float expm1f(float x)``
        | Returns :math:`ln(x - 1)`
      - ✓
      - ✓

    * - | ``float fabsf(float x)``
        | Returns the absolute value of `x`
      - ✓
      - ✓

    * - | ``float fdimf(float x, float y)``
        | Returns the positive difference between :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float fdividef(float x, float y)``
        | Divide two floating point values.
      - ✓
      - ✓

    * - | ``float floorf(float x)``
        | Returns the largest integer less than or equal to :math:`x`.
      - ✓
      - ✓

    * - | ``float fmaf(float x, float y, float z)``
        | Returns :math:`x \cdot y + z` as a single operation.
      - ✓
      - ✓

    * - | ``float fmaxf(float x, float y)``
        | Determine the maximum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float fminf(float x, float y)``
        | Determine the minimum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``float fmodf(float x, float y)``
        | Returns the floating-point remainder of :math:`x / y`.
      - ✓
      - ✓

    * - | ``float modff(float x, float* iptr)``
        | Break down :math:`x` into fractional and integral parts.
      - ✓
      - ✗

    * - | ``float frexpf(float x, int* nptr)``
        | Extract mantissa and exponent of :math:`x`.
      - ✓
      - ✗

    * - | ``float hypotf(float x, float y)``
        | Returns the square root of the sum of squares of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``int ilogbf(float x)``
        | Returns the unbiased integer exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``bool isfinite(float x)``
        | Determine whether :math:`x` is finite.
      - ✓
      - ✓

    * - | ``bool isinf(float x)``
        | Determine whether :math:`x` is infinite.
      - ✓
      - ✓

    * - | ``bool isnan(float x)``
        | Determine whether :math:`x` is a ``NAN``.
      - ✓
      - ✓

    * - | ``float j0f(float x)``
        | Returns the value of the Bessel function of the first kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``float j1f(float x)``
        | Returns the value of the Bessel function of the first kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``float jnf(int n, float x)``
        | Returns the value of the Bessel function of the first kind of order n for :math:`x`.
      - ✓
      - ✓

    * - | ``float ldexpf(float x, int exp)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``float lgammaf(float x)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✗

    * - | ``long int lrintf(float x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llrintf(float x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long int lroundf(float x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llroundf(float x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``float log10f(float x)``
        | Returns the base 10 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``float log1pf(float x)``
        | Returns the natural logarithm of :math:`x + 1`.
      - ✓
      - ✓

    * - | ``float log2f(float x)``
        | Returns the base 2 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``float logf(float x)``
        | Returns the natural logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``float logbf(float x)``
        | Returns the floating point representation of the exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``float nanf(const char* tagp)``
        | Returns "Not a Number" value.
      - ✗
      - ✓

    * - | ``float nearbyintf(float x)``
        | Round :math:`x` to the nearest integer.
      - ✓
      - ✓

    * - | ``float nextafterf(float x, float y)``
        | Returns next representable single-precision floating-point value after argument.
      - ✓
      - ✗

    * - | ``float norm3df(float x, float y, float z)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y` and :math:`z`.
      - ✓
      - ✓

    * - | ``float norm4df(float x, float y, float z, float w)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y`, :math:`z` and :math:`w`.
      - ✓
      - ✓

    * - | ``float normcdff(float y)``
        | Returns the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``float normcdfinvf(float y)``
        | Returns the inverse of the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``float normf(int dim, const float *a)``
        | Returns the square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``float powf(float x, float y)``
        | Returns :math:`x^y`.
      - ✓
      - ✓

    * - | ``float powif(float base, int iexp)``
        | Returns the value of first argument to the power of second argument.
      - ✓
      - ✓

    * - | ``float remainderf(float x, float y)``
        | Returns single-precision floating-point remainder.
      - ✓
      - ✓

    * - | ``float remquof(float x, float y, int* quo)``
        | Returns single-precision floating-point remainder and part of quotient.
      - ✓
      - ✓

    * - | ``float roundf(float x)``
        | Round to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``float rcbrtf(float x)``
        | Returns the reciprocal cube root function.
      - ✓
      - ✓

    * - | ``float rhypotf(float x, float y)``
        | Returns one over the square root of the sum of squares of two arguments.
      - ✓
      - ✓

    * - | ``float rintf(float x)``
        | Round input to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``float rnorm3df(float x, float y, float z)``
        | Returns one over the square root of the sum of squares of three coordinates of the argument.
      - ✓
      - ✓

    * - | ``float rnorm4df(float x, float y, float z, float w)``
        | Returns one over the square root of the sum of squares of four coordinates of the argument.
      - ✓
      - ✓

    * - | ``float rnormf(int dim, const float *a)``
        | Returns the reciprocal of square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``float scalblnf(float x, long int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``float scalbnf(float x, int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``bool signbit(float x)``
        | Return the sign bit of :math:`x`.
      - ✓
      - ✓

    * - | ``float sinf(float x)``
        | Returns the sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float sinhf(float x)``
        | Returns the hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``float sinpif(float x)``
        | Returns the hyperbolic sine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``void sincosf(float x, float *sptr, float *cptr)``
        | Returns the sine and cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``void sincospif(float x, float *sptr, float *cptr)``
        | Returns the sine and cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``float sqrtf(float x)``
        | Returns the square root of :math:`x`.
      - ✓
      - ✓

    * - | ``float rsqrtf(float x)``
        | Returns the reciprocal of the square root of :math:`x`.
      - ✗
      - ✓

    * - | ``float tanf(float x)``
        | Returns the tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float tanhf(float x)``
        | Returns the hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``float tgammaf(float x)``
        | Returns the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``float truncf(float x)``
        | Truncate :math:`x` to the integral part.
      - ✓
      - ✓

    * - | ``float y0f(float x)``
        | Returns the value of the Bessel function of the second kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``float y1f(float x)``
        | Returns the value of the Bessel function of the second kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``float ynf(int n, float x)``
        | Returns the value of the Bessel function of the second kind of order n for :math:`x`.
      - ✓
      - ✓

Double precision mathematical functions
=======================================

Following is the list of supported double precision mathematical functions.

.. list-table:: Double precision mathematical functions

    * - **Function**
      - **Supported on Host**
      - **Supported on Device**

    * - | ``double abs(double x)``
        | Returns the absolute value of :math:`x`
      - ✓
      - ✓

    * - | ``double acos(double x)``
        | Returns the arc cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double acosh(double x)``
        | Returns the nonnegative arc hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double asin(double x)``
        | Returns the arc sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double asinh(double x)``
        | Returns the arc hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double atan(double x)``
        | Returns the arc tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double atan2(double x, double y)``
        | Returns the arc tangent of the ratio of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double atanh(double x)``
        | Returns the arc hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double cbrt(double x)``
        | Returns the cube root of :math:`x`.
      - ✓
      - ✓

    * - | ``double ceil(double x)``
        | Returns ceiling of :math:`x`.
      - ✓
      - ✓

    * - | ``double copysign(double x, double y)``
        | Create value with given magnitude, copying sign of second value.
      - ✓
      - ✓

    * - | ``double cos(double x)``
        | Returns the cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double cosh(double x)``
        | Returns the hyperbolic cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``double cospi(double x)``
        | Returns the cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``double cyl_bessel_i0(double x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 0 for :math:`x`.
      - ✗
      - ✗

    * - | ``double cyl_bessel_i1(double x)``
        | Returns the value of the regular modified cylindrical Bessel function of order 1 for :math:`x`.
      - ✗
      - ✗

    * - | ``double erf(double x)``
        | Returns the error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfc(double x)``
        | Returns the complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfcinv(double x)``
        | Returns the inverse complementary function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfcx(double x)``
        | Returns the scaled complementary error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double erfinv(double x)``
        | Returns the inverse error function of :math:`x`.
      - ✓
      - ✓

    * - | ``double exp(double x)``
        | Returns :math:`e^x`.
      - ✓
      - ✓

    * - | ``double exp10(double x)``
        | Returns :math:`10^x`.
      - ✓
      - ✓

    * - | ``double exp2( double x)``
        | Returns :math:`2^x`.
      - ✓
      - ✓

    * - | ``double expm1(double x)``
        | Returns :math:`ln(x - 1)`
      - ✓
      - ✓

    * - | ``double fabs(double x)``
        | Returns the absolute value of `x`
      - ✓
      - ✓

    * - | ``double fdim(double x, double y)``
        | Returns the positive difference between :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double floor(double x)``
        | Returns the largest integer less than or equal to :math:`x`.
      - ✓
      - ✓

    * - | ``double fma(double x, double y, double z)``
        | Returns :math:`x \cdot y + z` as a single operation.
      - ✓
      - ✓

    * - | ``double fmax(double x, double y)``
        | Determine the maximum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double fmin(double x, double y)``
        | Determine the minimum numeric value of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``double fmod(double x, double y)``
        | Returns the floating-point remainder of :math:`x / y`.
      - ✓
      - ✓

    * - | ``double modf(double x, double* iptr)``
        | Break down :math:`x` into fractional and integral parts.
      - ✓
      - ✗

    * - | ``double frexp(double x, int* nptr)``
        | Extract mantissa and exponent of :math:`x`.
      - ✓
      - ✗

    * - | ``double hypot(double x, double y)``
        | Returns the square root of the sum of squares of :math:`x` and :math:`y`.
      - ✓
      - ✓

    * - | ``int ilogb(double x)``
        | Returns the unbiased integer exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``bool isfinite(double x)``
        | Determine whether :math:`x` is finite.
      - ✓
      - ✓

    * - | ``bool isin(double x)``
        | Determine whether :math:`x` is infinite.
      - ✓
      - ✓

    * - | ``bool isnan(double x)``
        | Determine whether :math:`x` is a ``NAN``.
      - ✓
      - ✓

    * - | ``double j0(double x)``
        | Returns the value of the Bessel function of the first kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``double j1(double x)``
        | Returns the value of the Bessel function of the first kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``double jn(int n, double x)``
        | Returns the value of the Bessel function of the first kind of order n for :math:`x`.
      - ✓
      - ✓

    * - | ``double ldexp(double x, int exp)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``double lgamma(double x)``
        | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
      - ✓
      - ✗

    * - | ``long int lrint(double x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llrint(double x)``
        | Round :math:`x` to nearest integer value.
      - ✓
      - ✓

    * - | ``long int lround(double x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``long long int llround(double x)``
        | Round to nearest integer value.
      - ✓
      - ✓

    * - | ``double log10(double x)``
        | Returns the base 10 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``double log1p(double x)``
        | Returns the natural logarithm of :math:`x + 1`.
      - ✓
      - ✓

    * - | ``double log2(double x)``
        | Returns the base 2 logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``double log(double x)``
        | Returns the natural logarithm of :math:`x`.
      - ✓
      - ✓

    * - | ``double logb(double x)``
        | Returns the floating point representation of the exponent of :math:`x`.
      - ✓
      - ✓

    * - | ``double nan(const char* tagp)``
        | Returns "Not a Number" value.
      - ✗
      - ✓

    * - | ``double nearbyint(double x)``
        | Round :math:`x` to the nearest integer.
      - ✓
      - ✓

    * - | ``double nextafter(double x, double y)``
        | Returns next representable double-precision floating-point value after argument.
      - ✓
      - ✓

    * - | ``double norm3d(double x, double y, double z)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y` and :math:`z`.
      - ✓
      - ✓

    * - | ``double norm4d(double x, double y, double z, double w)``
        | Returns the square root of the sum of squares of :math:`x`, :math:`y`, :math:`z` and :math:`w`.
      - ✓
      - ✓

    * - | ``double normcdf(double y)``
        | Returns the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``double normcdfinv(double y)``
        | Returns the inverse of the standard normal cumulative distribution function.
      - ✓
      - ✓

    * - | ``double norm(int dim, const double *a)``
        | Returns the square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``double pow(double x, double y)``
        | Returns :math:`x^y`.
      - ✓
      - ✓

    * - | ``double powi(double base, int iexp)``
        | Returns the value of first argument to the power of second argument.
      - ✓
      - ✓

    * - | ``double remainder(double x, double y)``
        | Returns double-precision floating-point remainder.
      - ✓
      - ✓

    * - | ``double remquo(double x, double y, int* quo)``
        | Returns double-precision floating-point remainder and part of quotient.
      - ✓
      - ✗

    * - | ``double round(double x)``
        | Round to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``double rcbrt(double x)``
        | Returns the reciprocal cube root function.
      - ✓
      - ✓

    * - | ``double rhypot(double x, double y)``
        | Returns one over the square root of the sum of squares of two arguments.
      - ✓
      - ✓

    * - | ``double rint(double x)``
        | Round input to nearest integer value in floating-point.
      - ✓
      - ✓

    * - | ``double rnorm3d(double x, double y, double z)``
        | Returns one over the square root of the sum of squares of three coordinates of the argument.
      - ✓
      - ✓

    * - | ``double rnorm4d(double x, double y, double z, double w)``
        | Returns one over the square root of the sum of squares of four coordinates of the argument.
      - ✓
      - ✓

    * - | ``double rnorm(int dim, const double *a)``
        | Returns the reciprocal of square root of the sum of squares of any number of coordinates.
      - ✓
      - ✓

    * - | ``double scalbln(double x, long int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``double scalbn(double x, int n)``
        | Scale :math:`x` by :math:`2^n`.
      - ✓
      - ✓

    * - | ``bool signbit(double x)``
        | Return the sign bit of :math:`x`.
      - ✓
      - ✓

    * - | ``double sin(double x)``
        | Returns the sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double sinh(double x)``
        | Returns the hyperbolic sine of :math:`x`.
      - ✓
      - ✓

    * - | ``double sinpi(double x)``
        | Returns the hyperbolic sine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``void sincos(double x, double *sptr, double *cptr)``
        | Returns the sine and cosine of :math:`x`.
      - ✓
      - ✓

    * - | ``void sincospi(double x, double *sptr, double *cptr)``
        | Returns the sine and cosine of :math:`\pi \cdot x`.
      - ✓
      - ✓

    * - | ``double sqrt(double x)``
        | Returns the square root of :math:`x`.
      - ✓
      - ✓

    * - | ``double rsqrt(double x)``
        | Returns the reciprocal of the square root of :math:`x`.
      - ✗
      - ✓

    * - | ``double tan(double x)``
        | Returns the tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double tanh(double x)``
        | Returns the hyperbolic tangent of :math:`x`.
      - ✓
      - ✓

    * - | ``double tgamma(double x)``
        | Returns the gamma function of :math:`x`.
      - ✓
      - ✓

    * - | ``double trunc(double x)``
        | Truncate :math:`x` to the integral part.
      - ✓
      - ✓

    * - | ``double y0(double x)``
        | Returns the value of the Bessel function of the second kind of order 0 for :math:`x`.
      - ✓
      - ✓

    * - | ``double y1(double x)``
        | Returns the value of the Bessel function of the second kind of order 1 for :math:`x`.
      - ✓
      - ✓

    * - | ``double yn(int n, double x)``
        | Returns the value of the Bessel function of the second kind of order n for :math:`x`.
      - ✓
      - ✓

Integer intrinsics
==================

Following is the list of supported integer intrinsics. Note that intrinsics are supported on device only.

.. list-table:: Integer intrinsics mathematical functions

    * - **Function**

    * - | ``unsigned int __brev(unsigned int x)``
        | Reverse the bit order of a 32 bit unsigned integer.

    * - | ``unsigned long long int __brevll(unsigned long long int x)``
        | Reverse the bit order of a 64 bit unsigned integer.

    * - | ``unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int z)``
        | Return selected bytes from two 32-bit unsigned integers.

    * - | ``unsigned int __clz(int x)``
        | Return the number of consecutive high-order zero bits in 32 bit integer.

    * - | ``unsigned int __clzll(long long int x)``
        | Return the number of consecutive high-order zero bits in 64 bit integer.

    * - | ``unsigned int __ffs(int x)``
        | Find the position of least significant bit set to 1 in a 32 bit integer.

    * - | ``unsigned int __ffsll(long long int x)``
        | Find the position of least significant bit set to 1 in a 64 bit signed integer.

    * - | ``unsigned int __fns32(unsigned long long mask, unsigned int base, int offset)``
        | Find the position of the n-th set to 1 bit in a 32-bit integer.

    * - | ``unsigned int __fns64(unsigned long long int mask, unsigned int base, int offset)``
        | Find the position of the n-th set to 1 bit in a 64-bit integer.

    * - | ``unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift left by shift & 31 bits, return the most significant 32 bits.

    * - | ``unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift left by min(shift, 32) bits, return the most significant 32 bits.

    * - | ``unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift right by shift & 31 bits, return the least significant 32 bits.

    * - | ``unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)``
        | Concatenate :math:`hi` and :math:`lo`, shift right by min(shift, 32) bits, return the least significant 32 bits.

    * - | ``unsigned int __hadd(int x, int y)``
        | Compute average of signed input arguments, avoiding overflow in the intermediate sum.

    * - | ``unsigned int __rhadd(int x, int y)``
        | Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum.

    * - | ``unsigned int __uhadd(int x, int y)``
        | Compute average of unsigned input arguments, avoiding overflow in the intermediate sum.

    * - | ``unsigned int __urhadd (unsigned int x, unsigned int y)``
        | Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum.

    * - | ``int __sad(int x, int y, int z)``
        | Returns :math:`|x - y| + z`, the sum of absolute difference.

    * - | ``unsigned int __usad(unsigned int x, unsigned int y, unsigned int z)``
        | Returns :math:`|x - y| + z`, the sum of absolute difference.

    * - | ``unsigned int __popc(unsigned int x)``
        | Count the number of bits that are set to 1 in a 32 bit integer.

    * - | ``unsigned int __popcll(unsigned long long int x)``
        | Count the number of bits that are set to 1 in a 64 bit integer.

    * - | ``int __mul24(int x, int y)``
        | Multiply two 24bit integers.

    * - | ``unsigned int __umul24(unsigned int x, unsigned int y)``
        | Multiply two 24bit unsigned integers.

    * - | ``int __mulhi(int x, int y)``
        | Returns the most significant 32 bits of the product of the two 32-bit integers.

    * - | ``unsigned int __umulhi(unsigned int x, unsigned int y)``
        | Returns the most significant 32 bits of the product of the two 32-bit unsigned integers.

    * - | ``long long int __mul64hi(long long int x, long long int y)``
        | Returns the most significant 64 bits of the product of the two 64-bit integers.

    * - | ``unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y)``
        | Returns the most significant 64 bits of the product of the two 64 unsigned bit integers.

The HIP-Clang implementation of ``__ffs()`` and ``__ffsll()`` contains code to add a constant +1 to produce the ``ffs`` result format.
For the cases where this overhead is not acceptable and programmer is willing to specialize for the platform,
HIP-Clang provides ``__lastbit_u32_u32(unsigned int input)`` and ``__lastbit_u32_u64(unsigned long long int input)``.
The index returned by ``__lastbit_`` instructions starts at -1, while for ``ffs`` the index starts at 0.

Floating-point Intrinsics
=========================

Following is the list of supported floating-point intrinsics. Note that intrinsics are supported on device only.

.. note::

  Only the nearest even rounding mode supported on AMD GPUs by defaults. The ``_rz``, ``_ru`` and
  ``_rd`` suffixed intrinsic functions are existing in HIP AMD backend, if the
  ``OCML_BASIC_ROUNDED_OPERATIONS`` macro is defined.

.. list-table:: Single precision intrinsics mathematical functions

    * - **Function**

    * - | ``float __cosf(float x)``
        | Returns the fast approximate cosine of :math:`x`.

    * - | ``float __exp10f(float x)``
        | Returns the fast approximate for 10 :sup:`x`.

    * - | ``float __expf(float x)``
        | Returns the fast approximate for e :sup:`x`.

    * - | ``float __fadd_rn(float x, float y)``
        | Add two floating-point values in round-to-nearest-even mode.

    * - | ``float __fdiv_rn(float x, float y)``
        | Divide two floating point values in round-to-nearest-even mode.

    * - | ``float __fmaf_rn(float x, float y, float z)``
        | Returns ``x × y + z`` as a single operation in round-to-nearest-even mode.

    * - | ``float __fmul_rn(float x, float y)``
        | Multiply two floating-point values in round-to-nearest-even mode.

    * - | ``float __frcp_rn(float x, float y)``
        | Returns ``1 / x`` in round-to-nearest-even mode.

    * - | ``float __frsqrt_rn(float x)``
        | Returns ``1 / √x`` in round-to-nearest-even mode.

    * - | ``float __fsqrt_rn(float x)``
        | Returns ``√x`` in round-to-nearest-even mode.

    * - | ``float __fsub_rn(float x, float y)``
        | Subtract two floating-point values in round-to-nearest-even mode.

    * - | ``float __log10f(float x)``
        | Returns the fast approximate for base 10 logarithm of :math:`x`.

    * - | ``float __log2f(float x)``
        | Returns the fast approximate for base 2 logarithm of :math:`x`.

    * - | ``float __logf(float x)``
        | Returns the fast approximate for natural logarithm of :math:`x`.

    * - | ``float __powf(float x, float y)``
        | Returns the fast approximate of x :sup:`y`.

    * - | ``float __saturatef(float x)``
        | Clamp :math:`x` to [+0.0, 1.0].

    * - | ``float __sincosf(float x, float* sinptr, float* cosptr)``
        | Returns the fast approximate of sine and cosine of :math:`x`.

    * - | ``float __sinf(float x)``
        | Returns the fast approximate sine of :math:`x`.

    * - | ``float __tanf(float x)``
        | Returns the fast approximate tangent of :math:`x`.

.. list-table:: Double precision intrinsics mathematical functions

    * - **Function**

    * - | ``double __dadd_rn(double x, double y)``
        | Add two floating-point values in round-to-nearest-even mode.

    * - | ``double __ddiv_rn(double x, double y)``
        | Divide two floating-point values in round-to-nearest-even mode.

    * - | ``double __dmul_rn(double x, double y)``
        | Multiply two floating-point values in round-to-nearest-even mode.

    * - | ``double __drcp_rn(double x, double y)``
        | Returns ``1 / x`` in round-to-nearest-even mode.

    * - | ``double __dsqrt_rn(double x)``
        | Returns ``√x`` in round-to-nearest-even mode.

    * - | ``double __dsub_rn(double x, double y)``
        | Subtract two floating-point values in round-to-nearest-even mode.

    * - | ``double __fma_rn(double x, double y, double z)``
        | Returns ``x × y + z`` as a single operation in round-to-nearest-even mode.
