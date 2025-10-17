.. meta::
  :description: This chapter describes the built-in math functions that are accessible in HIP.
  :keywords: AMD, ROCm, HIP, CUDA, math functions, HIP math functions

.. _math_api_reference:

********************************************************************************
HIP math API
********************************************************************************

HIP-Clang provides device-callable math operations, supporting most functions available in
NVIDIA CUDA.

This section documents:

- Maximum error bounds for supported HIP math functions
- Currently unsupported functions

Error bounds on this page are measured in units in the last place (ULPs), representing the absolute
difference between a HIP math function result and its corresponding C++ standard library function
(e.g., comparing HIP's sinf with C++'s sinf).

The following C++ example shows a simplified method for computing ULP differences between
HIP and standard C++ math functions by first finding where the maximum absolute error
occurs.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <limits>

    #define HIP_CHECK(expression)              \
        {                                      \
            const hipError_t err = expression; \
            if (err != hipSuccess) {           \
                std::cerr << "HIP error: "     \
                          << hipGetErrorString(err) \
                          << " at " << __LINE__ << "\n"; \
                exit(EXIT_FAILURE);            \
            }                                  \
        }

    // Simple ULP difference calculator
    int64_t ulp_diff(float a, float b) {
        if (a == b) return 0;
        union { float f; int32_t i; } ua{a}, ub{b};

        // For negative values, convert to a positive-based representation
        if (ua.i < 0) ua.i = std::numeric_limits<int32_t>::max() - ua.i;
        if (ub.i < 0) ub.i = std::numeric_limits<int32_t>::max() - ub.i;

        return std::abs((int64_t)ua.i - (int64_t)ub.i);
    }

    // Test kernel
    __global__ void test_sin(float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float x = -M_PI + (2.0f * M_PI * i) / (n - 1);
            out[i] = sin(x);
        }
    }

    int main() {
        const int n = 1000000;
        const int blocksize = 256;
        std::vector<float> outputs(n);
        float* d_out;

        HIP_CHECK(hipMalloc(&d_out, n * sizeof(float)));
        dim3 threads(blocksize);
        dim3 blocks((n + blocksize - 1) / blocksize);  // Fixed grid calculation
        test_sin<<<blocks, threads>>>(d_out, n);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipMemcpy(outputs.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost));

        // Step 1: Find the maximum absolute error
        double max_abs_error = 0.0;
        float max_error_output = 0.0;
        float max_error_expected = 0.0;

        for (int i = 0; i < n; i++) {
            float x = -M_PI + (2.0f * M_PI * i) / (n - 1);
            float expected = std::sin(x);
            double abs_error = std::abs(outputs[i] - expected);

            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
                max_error_output = outputs[i];
                max_error_expected = expected;
            }
        }

        // Step 2: Compute ULP difference based on the max absolute error pair
        int64_t max_ulp = ulp_diff(max_error_output, max_error_expected);

        // Output results
        std::cout << "Max Absolute Error: " << max_abs_error << std::endl;
        std::cout << "Max ULP Difference: " << max_ulp << std::endl;
        std::cout << "Max Error Values -> Got: " << max_error_output
                  << ", Expected: " << max_error_expected << std::endl;

        HIP_CHECK(hipFree(d_out));
        return 0;
    }

Standard mathematical functions
===============================

The functions in this section prioritize numerical accuracy and correctness, making them well-suited for
applications that require high precision and predictable results. Unless explicitly specified, all
math functions listed below are available on the device side.

Arithmetic
----------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float abs(float x)``
          | Returns the absolute value of :math:`x`
        - :math:`x \in [-20, 20]`
        - 0

      * - | ``float fabsf(float x)``
          | Returns the absolute value of `x`
        - :math:`x \in [-20, 20]`
        - 0

      * - | ``float fdimf(float x, float y)``
          | Returns the positive difference between :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float fmaf(float x, float y, float z)``
          | Returns :math:`x \cdot y + z` as a single operation.
        - | :math:`x \in [-100, 100]`
          | :math:`y \in [-10, 10]`
          | :math:`z \in [-10, 10]`
        - 0

      * - | ``float fmaxf(float x, float y)``
          | Determine the maximum numeric value of :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float fminf(float x, float y)``
          | Determine the minimum numeric value of :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float fmodf(float x, float y)``
          | Returns the floating-point remainder of :math:`x / y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float modff(float x, float* iptr)``
          | Break down :math:`x` into fractional and integral parts.
        - :math:`x \in [-10, 10]`
        - 0

      * - | ``float remainderf(float x, float y)``
          | Returns single-precision floating-point remainder.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float remquof(float x, float y, int* quo)``
          | Returns single-precision floating-point remainder and part of quotient.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float fdividef(float x, float y)``
          | Divide two floating point values.
        - | :math:`x \in [-100, 100]`
          | :math:`y \in [-100, 100]`
        - 0


  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double abs(double x)``
          | Returns the absolute value of :math:`x`
        - :math:`x \in [-20, 20]`
        - 0

      * - | ``double fabs(double x)``
          | Returns the absolute value of `x`
        - :math:`x \in [-20, 20]`
        - 0

      * - | ``double fdim(double x, double y)``
          | Returns the positive difference between :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``double fma(double x, double y, double z)``
          | Returns :math:`x \cdot y + z` as a single operation.
        - | :math:`x \in [-100, 100]`
          | :math:`y \in [-10, 10]`
          | :math:`z \in [-10, 10]`
        - 0

      * - | ``double fmax(double x, double y)``
          | Determine the maximum numeric value of :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``double fmin(double x, double y)``
          | Determine the minimum numeric value of :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``double fmod(double x, double y)``
          | Returns the floating-point remainder of :math:`x / y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``double modf(double x, double* iptr)``
          | Break down :math:`x` into fractional and integral parts.
        - :math:`x \in [-10, 10]`
        - 0

      * - | ``double remainder(double x, double y)``
          | Returns double-precision floating-point remainder.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``double remquo(double x, double y, int* quo)``
          | Returns double-precision floating-point remainder and part of quotient.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

Classification
--------------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``bool isfinite(float x)``
          | Determine whether :math:`x` is finite.
        - | :math:`x \in [-\text{FLT_MAX}, \text{FLT_MAX}]`
          | Special values: :math:`\pm\infty`, NaN
        - 0

      * - | ``bool isinf(float x)``
          | Determine whether :math:`x` is infinite.
        - | :math:`x \in [-\text{FLT_MAX}, \text{FLT_MAX}]`
          | Special values: :math:`\pm\infty`, NaN
        - 0

      * - | ``bool isnan(float x)``
          | Determine whether :math:`x` is a ``NAN``.
        - | :math:`x \in [-\text{FLT_MAX}, \text{FLT_MAX}]`
          | Special values: :math:`\pm\infty`, NaN
        - 0

      * - | ``bool signbit(float x)``
          | Return the sign bit of :math:`x`.
        - | :math:`x \in [-\text{FLT_MAX}, \text{FLT_MAX}]`
          | Special values: :math:`\pm\infty`, :math:`\pm0`, NaN
        - 0

      * - | ``float nanf(const char* tagp)``
          | Returns "Not a Number" value.
        - | Input strings: ``""``, ``"1"``, ``"2"``,
          | ``"quiet"``, ``"signaling"``, ``"ind"``
        - 0

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``bool isfinite(double x)``
          | Determine whether :math:`x` is finite.
        - | :math:`x \in [-\text{DBL_MAX}, \text{DBL_MAX}]`
          | Special values: :math:`\pm\infty`, NaN
        - 0

      * - | ``bool isin(double x)``
          | Determine whether :math:`x` is infinite.
        - | :math:`x \in [-\text{DBL_MAX}, \text{DBL_MAX}]`
          | Special values: :math:`\pm\infty`, NaN
        - 0

      * - | ``bool isnan(double x)``
          | Determine whether :math:`x` is a ``NAN``.
        - | :math:`x \in [-\text{DBL_MAX}, \text{DBL_MAX}]`
          | Special values: :math:`\pm\infty`, NaN
        - 0

      * - | ``bool signbit(double x)``
          | Return the sign bit of :math:`x`.
        - | :math:`x \in [-\text{DBL_MAX}, \text{DBL_MAX}]`
          | Special values: :math:`\pm\infty`, :math:`\pm0`, NaN
        - 0

      * - | ``double nan(const char* tagp)``
          | Returns "Not a Number" value.
        - | Input strings: ``""``, ``"1"``, ``"2"``,
          | ``"quiet"``, ``"signaling"``, ``"ind"``
        - 0

Error and Gamma
---------------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float erff(float x)``
          | Returns the error function of :math:`x`.
        - :math:`x \in [-4, 4]`
        - 4

      * - | ``float erfcf(float x)``
          | Returns the complementary error function of :math:`x`.
        - :math:`x \in [-4, 4]`
        - 2

      * - | ``float erfcxf(float x)``
          | Returns the scaled complementary error function of :math:`x`.
        - :math:`x \in [-2, 2]`
        - 5

      * - | ``float lgammaf(float x)``
          | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
        - :math:`x \in [0.5, 20]`
        - 4

      * - | ``float tgammaf(float x)``
          | Returns the gamma function of :math:`x`.
        - :math:`x \in [0.5, 15]`
        - 6

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double erf(double x)``
          | Returns the error function of :math:`x`.
        - :math:`x \in [-4, 4]`
        - 4

      * - | ``double erfc(double x)``
          | Returns the complementary error function of :math:`x`.
        - :math:`x \in [-4, 4]`
        - 2

      * - | ``double erfcx(double x)``
          | Returns the scaled complementary error function of :math:`x`.
        - :math:`x \in [-2, 2]`
        - 5

      * - | ``double lgamma(double x)``
          | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
        - :math:`x \in [0.5, 20]`
        - 2

      * - | ``double tgamma(double x)``
          | Returns the gamma function of :math:`x`.
        - :math:`x \in [0.5, 15]`
        - 6

Exponential and Logarithmic
---------------------------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float expf(float x)``
          | Returns :math:`e^x`.
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``float exp2f(float x)``
          | Returns :math:`2^x`.
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``float exp10f(float x)``
          | Returns :math:`10^x`.
        - :math:`x \in [-4, 4]`
        - 1

      * - | ``float expm1f(float x)``
          | Returns :math:`ln(x - 1)`
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``float log10f(float x)``
          | Returns the base 10 logarithm of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 2

      * - | ``float log1pf(float x)``
          | Returns the natural logarithm of :math:`x + 1`.
        - :math:`x \in [-0.9, 10]`
        - 1

      * - | ``float log2f(float x)``
          | Returns the base 2 logarithm of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 1

      * - | ``float logf(float x)``
          | Returns the natural logarithm of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 2

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double exp(double x)``
          | Returns :math:`e^x`.
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``double exp2(double x)``
          | Returns :math:`2^x`.
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``double exp10(double x)``
          | Returns :math:`10^x`.
        - :math:`x \in [-4, 4]`
        - 1

      * - | ``double expm1(double x)``
          | Returns :math:`ln(x - 1)`
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``double log10(double x)``
          | Returns the base 10 logarithm of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 1

      * - | ``double log1p(double x)``
          | Returns the natural logarithm of :math:`x + 1`.
        - :math:`x \in [-0.9, 10]`
        - 1

      * - | ``double log2(double x)``
          | Returns the base 2 logarithm of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 1

      * - | ``double log(double x)``
          | Returns the natural logarithm of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 1

Floating Point Manipulation
---------------------------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float copysignf(float x, float y)``
          | Create value with given magnitude, copying sign of second value.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float frexpf(float x, int* nptr)``
          | Extract mantissa and exponent of :math:`x`.
        - :math:`x \in [-10, 10]`
        - 0

      * - | ``int ilogbf(float x)``
          | Returns the unbiased integer exponent of :math:`x`.
        - :math:`x \in [0.01, 100]`
        - 0

      * - | ``float logbf(float x)``
          | Returns the floating point representation of the exponent of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 0

      * - | ``float ldexpf(float x, int exp)``
          | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
        - | :math:`x \in [-10, 10]`
          | :math:`\text{exp} \in [-4, 4]`
        - 0

      * - | ``float nextafterf(float x, float y)``
          | Returns next representable single-precision floating-point value after argument.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``float scalblnf(float x, long int n)``
          | Scale :math:`x` by :math:`2^n`.
        - | :math:`x \in [-10, 10]`
          | :math:`n \in [-4, 4]`
        - 0

      * - | ``float scalbnf(float x, int n)``
          | Scale :math:`x` by :math:`2^n`.
        - | :math:`x \in [-10, 10]`
          | :math:`n \in [-4, 4]`
        - 0

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double copysign(double x, double y)``
          | Create value with given magnitude, copying sign of second value.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``double frexp(double x, int* nptr)``
          | Extract mantissa and exponent of :math:`x`.
        - :math:`x \in [-10, 10]`
        - 0

      * - | ``int ilogb(double x)``
          | Returns the unbiased integer exponent of :math:`x`.
        - :math:`x \in [0.01, 100]`
        - 0

      * - | ``double logb(double x)``
          | Returns the floating point representation of the exponent of :math:`x`.
        - :math:`x \in [10^{-6}, 10^6]`
        - 0

      * - | ``double ldexp(double x, int exp)``
          | Returns the natural logarithm of the absolute value of the gamma function of :math:`x`.
        - | :math:`x \in [-10, 10]`
          | :math:`\text{exp} \in [-4, 4]`
        - 0

      * - | ``double nextafter(double x, double y)``
          | Returns next representable double-precision floating-point value after argument.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [-3, 3]`
        - 0

      * - | ``double scalbln(double x, long int n)``
          | Scale :math:`x` by :math:`2^n`.
        - | :math:`x \in [-10, 10]`
          | :math:`n \in [-4, 4]`
        - 0

      * - | ``double scalbn(double x, int n)``
          | Scale :math:`x` by :math:`2^n`.
        - | :math:`x \in [-10, 10]`
          | :math:`n \in [-4, 4]`
        - 0

Hypotenuse and Norm
-------------------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float hypotf(float x, float y)``
          | Returns the square root of the sum of squares of :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [0, 10]`
        - 1

      * - | ``float rhypotf(float x, float y)``
          | Returns one over the square root of the sum of squares of two arguments.
        - | :math:`x \in [-100, 100]`
          | :math:`y \in [-10, 100]`
        - 1

      * - | ``float norm3df(float x, float y, float z)``
          | Returns the square root of the sum of squares of :math:`x`, :math:`y` and :math:`z`.
        - | All inputs in
          | :math:`[-10, 10]`
        - 1

      * - | ``float norm4df(float x, float y, float z, float w)``
          | Returns the square root of the sum of squares of :math:`x`, :math:`y`, :math:`z` and :math:`w`.
        - | All inputs in
          | :math:`[-10, 10]`
        - 2

      * - | ``float rnorm3df(float x, float y, float z)``
          | Returns one over the square root of the sum of squares of three coordinates of the argument.
        - | All inputs in
          | :math:`[-10, 10]`
        - 1

      * - | ``float rnorm4df(float x, float y, float z, float w)``
          | Returns one over the square root of the sum of squares of four coordinates of the argument.
        - | All inputs in
          | :math:`[-10, 10]`
        - 2

      * - | ``float normf(int dim, const float *a)``
          | Returns the square root of the sum of squares of any number of coordinates.
        - | :math:`\text{dim} \in [2,4]`
          | :math:`a[i] \in [-10, 10]`
        - | Error depends on the number of coordinates
          | e.g. ``dim = 2`` -> 1
          | e.g. ``dim = 3`` -> 1
          | e.g. ``dim = 4`` -> 1

      * - | ``float rnormf(int dim, const float *a)``
          | Returns the reciprocal of square root of the sum of squares of any number of coordinates.
        - | :math:`\text{dim} \in [2,4]`
          | :math:`a[i] \in [-10, 10]`
        - | Error depends on the number of coordinates
          | e.g. ``dim = 2`` -> 1
          | e.g. ``dim = 3`` -> 1
          | e.g. ``dim = 4`` -> 1

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double hypot(double x, double y)``
          | Returns the square root of the sum of squares of :math:`x` and :math:`y`.
        - | :math:`x \in [-10, 10]`
          | :math:`y \in [0, 10]`
        - 1

      * - | ``double rhypot(double x, double y)``
          | Returns one over the square root of the sum of squares of two arguments.
        - | :math:`x \in [-100, 100]`
          | :math:`y \in [-10, 100]`
        - 1

      * - | ``double norm3d(double x, double y, double z)``
          | Returns the square root of the sum of squares of :math:`x`, :math:`y` and :math:`z`.
        - | All inputs in
          | :math:`[-10, 10]`
        - 1

      * - | ``double norm4d(double x, double y, double z, double w)``
          | Returns the square root of the sum of squares of :math:`x`, :math:`y`, :math:`z` and :math:`w`.
        - | All inputs in
          | :math:`[-10, 10]`
        - 2

      * - | ``double rnorm3d(double x, double y, double z)``
          | Returns one over the square root of the sum of squares of three coordinates of the argument.
        - | All inputs in
          | :math:`[-10, 10]`
        - 1

      * - | ``double rnorm4d(double x, double y, double z, double w)``
          | Returns one over the square root of the sum of squares of four coordinates of the argument.
        - | All inputs in
          | :math:`[-10, 10]`
        - 1

      * - | ``double norm(int dim, const double *a)``
          | Returns the square root of the sum of squares of any number of coordinates.
        - | :math:`\text{dim} \in [2,4]`
          | :math:`a[i] \in [-10, 10]`
        - | Error depends on the number of coordinates
          | e.g. ``dim = 2`` -> 1
          | e.g. ``dim = 3`` -> 1
          | e.g. ``dim = 4`` -> 1

      * - | ``double rnorm(int dim, const double *a)``
          | Returns the reciprocal of square root of the sum of squares of any number of coordinates.
        - | :math:`\text{dim} \in [2,4]`
          | :math:`a[i] \in [-10, 10]`
        - | Error depends on the number of coordinates
          | e.g. ``dim = 2`` -> 1
          | e.g. ``dim = 3`` -> 1
          | e.g. ``dim = 4`` -> 1


Power and Root
--------------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float cbrtf(float x)``
          | Returns the cube root of :math:`x`.
        - :math:`x \in [-100, 100]`
        - 2

      * - | ``float powf(float x, float y)``
          | Returns :math:`x^y`.
        - | :math:`x \in [-4, 4]`
          | :math:`y \in [-2, 2]`
        - 1

      * - | ``float powif(float base, int iexp)``
          | Returns the value of first argument to the power of second argument.
        - | :math:`\text{base} \in [-10, 10]`
          | :math:`\text{iexp} \in [-4, 4]`
        - 1

      * - | ``float sqrtf(float x)``
          | Returns the square root of :math:`x`.
        - :math:`x \in [0, 100]`
        - 1

      * - | ``float rsqrtf(float x)``
          | Returns the reciprocal of the square root of :math:`x`.
        - :math:`x \in [0.01, 100]`
        - 1

      * - | ``float rcbrtf(float x)``
          | Returns the reciprocal cube root function.
        - :math:`x \in [-100, 100]`
        - 1

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double cbrt(double x)``
          | Returns the cube root of :math:`x`.
        - :math:`x \in [-100, 100]`
        - 1

      * - | ``double pow(double x, double y)``
          | Returns :math:`x^y`.
        - | :math:`x \in [-4, 4]`
          | :math:`y \in [-2, 2]`
        - 1

      * - | ``double powi(double base, int iexp)``
          | Returns the value of first argument to the power of second argument.
        - | :math:`\text{base} \in [-10, 10]`
          | :math:`\text{iexp} \in [-4, 4]`
        - 1

      * - | ``double sqrt(double x)``
          | Returns the square root of :math:`x`.
        - :math:`x \in [0, 100]`
        - 1

      * - | ``double rsqrt(double x)``
          | Returns the reciprocal of the square root of :math:`x`.
        - :math:`x \in [0.01, 100]`
        - 1

      * - | ``double rcbrt(double x)``
          | Returns the reciprocal cube root function.
        - :math:`x \in [-100, 100]`
        - 1

Rounding
--------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float ceilf(float x)``
          | Returns ceiling of :math:`x`.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``float floorf(float x)``
          | Returns the largest integer less than or equal to :math:`x`.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long int lroundf(float x)``
          | Round to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long long int llroundf(float x)``
          | Round to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long int lrintf(float x)``
          | Round :math:`x` to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long long int llrintf(float x)``
          | Round :math:`x` to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``float nearbyintf(float x)``
          | Round :math:`x` to the nearest integer.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``float roundf(float x)``
          | Round to nearest integer value in floating-point.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``float rintf(float x)``
          | Round input to nearest integer value in floating-point.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``float truncf(float x)``
          | Truncate :math:`x` to the integral part.
        - :math:`x \in [-4, 4]`
        - 0

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double ceil(double x)``
          | Returns ceiling of :math:`x`.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``double floor(double x)``
          | Returns the largest integer less than or equal to :math:`x`.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long int lround(double x)``
          | Round to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long long int llround(double x)``
          | Round to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long int lrint(double x)``
          | Round :math:`x` to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``long long int llrint(double x)``
          | Round :math:`x` to nearest integer value.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``double nearbyint(double x)``
          | Round :math:`x` to the nearest integer.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``double round(double x)``
          | Round to nearest integer value in floating-point.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``double rint(double x)``
          | Round input to nearest integer value in floating-point.
        - :math:`x \in [-4, 4]`
        - 0

      * - | ``double trunc(double x)``
          | Truncate :math:`x` to the integral part.
        - :math:`x \in [-4, 4]`
        - 0

Trigonometric and Hyperbolic
----------------------------
.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``float acosf(float x)``
          | Returns the arc cosine of :math:`x`.
        - :math:`x \in [-1, 1]`
        - 1

      * - | ``float acoshf(float x)``
          | Returns the nonnegative arc hyperbolic cosine of :math:`x`.
        - :math:`x \in [1, 100]`
        - 1

      * - | ``float asinf(float x)``
          | Returns the arc sine of :math:`x`.
        - :math:`x \in [-1, 1]`
        - 2

      * - | ``float asinhf(float x)``
          | Returns the arc hyperbolic sine of :math:`x`.
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``float atanf(float x)``
          | Returns the arc tangent of :math:`x`.
        - :math:`x \in [-10, 10]`
        - 2

      * - | ``float atan2f(float x, float y)``
          | Returns the arc tangent of the ratio of :math:`x` and :math:`y`.
        - | :math:`x \in [-4, 4]`
          | :math:`y \in [-2, 2]`
        - 1

      * - | ``float atanhf(float x)``
          | Returns the arc hyperbolic tangent of :math:`x`.
        - :math:`x \in [-0.9, 0.9]`
        - 1

      * - | ``float cosf(float x)``
          | Returns the cosine of :math:`x`.
        - :math:`x \in [-\pi, \pi]`
        - 1

      * - | ``float coshf(float x)``
          | Returns the hyperbolic cosine of :math:`x`.
        - :math:`x \in [-5, 5]`
        - 1

      * - | ``float sinf(float x)``
          | Returns the sine of :math:`x`.
        - :math:`x \in [-\pi, \pi]`
        - 1

      * - | ``float sinhf(float x)``
          | Returns the hyperbolic sine of :math:`x`.
        - :math:`x \in [-5, 5]`
        - 1

      * - | ``void sincosf(float x, float *sptr, float *cptr)``
          | Returns the sine and cosine of :math:`x`.
        - :math:`x \in [-3, 3]`
        - | ``sin``: 1
          | ``cos``: 1

      * - | ``float tanf(float x)``
          | Returns the tangent of :math:`x`.
        - :math:`x \in [-1.47\pi, 1.47\pi]`
        - 1

      * - | ``float tanhf(float x)``
          | Returns the hyperbolic tangent of :math:`x`.
        - :math:`x \in [-5, 5]`
        - 2

      * - | ``float cospif(float x)``
          | Returns the cosine of :math:`\pi \cdot x`.
        - :math:`x \in [-0.3, 0.3]`
        - 1

      * - | ``float sinpif(float x)``
          | Returns the hyperbolic sine of :math:`\pi \cdot x`.
        - :math:`x \in [-0.625, 0.625]`
        - 2

      * - | ``void sincospif(float x, float *sptr, float *cptr)``
          | Returns the sine and cosine of :math:`\pi \cdot x`.
        - :math:`x \in [-0.3, 0.3]`
        - | ``sinpi``: 2
          | ``cospi``: 1

  .. tab-item:: Double Precision Floating-point

    .. list-table::
      :widths: 50,20,30

      * - **Function**
        - **Test Range**
        - **ULP Difference of Maximum Absolute Error**

      * - | ``double acos(double x)``
          | Returns the arc cosine of :math:`x`.
        - :math:`x \in [-1, 1]`
        - 1

      * - | ``double acosh(double x)``
          | Returns the nonnegative arc hyperbolic cosine of :math:`x`.
        - :math:`x \in [1, 100]`
        - 1

      * - | ``double asin(double x)``
          | Returns the arc sine of :math:`x`.
        - :math:`x \in [-1, 1]`
        - 1

      * - | ``double asinh(double x)``
          | Returns the arc hyperbolic sine of :math:`x`.
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``double atan(double x)``
          | Returns the arc tangent of :math:`x`.
        - :math:`x \in [-10, 10]`
        - 1

      * - | ``double atan2(double x, double y)``
          | Returns the arc tangent of the ratio of :math:`x` and :math:`y`.
        - | :math:`x \in [-4, 4]`
          | :math:`y \in [-2, 2]`
        - 1

      * - | ``double atanh(double x)``
          | Returns the arc hyperbolic tangent of :math:`x`.
        - :math:`x \in [-0.9, 0.9]`
        - 1

      * - | ``double cos(double x)``
          | Returns the cosine of :math:`x`.
        - :math:`x \in [-\pi, \pi]`
        - 1

      * - | ``double cosh(double x)``
          | Returns the hyperbolic cosine of :math:`x`.
        - :math:`x \in [-5, 5]`
        - 1

      * - | ``double sin(double x)``
          | Returns the sine of :math:`x`.
        - :math:`x \in [-\pi, \pi]`
        - 1

      * - | ``double sinh(double x)``
          | Returns the hyperbolic sine of :math:`x`.
        - :math:`x \in [-5, 5]`
        - 1

      * - | ``void sincos(double x, double *sptr, double *cptr)``
          | Returns the sine and cosine of :math:`x`.
        - :math:`x \in [-3, 3]`
        - | ``sin``: 1
          | ``cos``: 1

      * - | ``double tan(double x)``
          | Returns the tangent of :math:`x`.
        - :math:`x \in [-1.47\pi, 1.47\pi]`
        - 1

      * - | ``double tanh(double x)``
          | Returns the hyperbolic tangent of :math:`x`.
        - :math:`x \in [-5, 5]`
        - 1

      * - | ``double cospi(double x)``
          | Returns the cosine of :math:`\pi \cdot x`.
        - :math:`x \in [-0.3, 0.3]`
        - 2

      * - | ``double sinpi(double x)``
          | Returns the hyperbolic sine of :math:`\pi \cdot x`.
        - :math:`x \in [-0.625, 0.625]`
        - 2

      * - | ``void sincospi(double x, double *sptr, double *cptr)``
          | Returns the sine and cosine of :math:`\pi \cdot x`.
        - :math:`x \in [-0.3, 0.3]`
        - | ``sinpi``: 2
          | ``cospi``: 2

No C++ STD Implementation
-------------------------

This table lists HIP device functions that do not have a direct equivalent in the C++ standard library.
These functions were excluded from comparison due to the complexity of implementing a precise
reference version within the standard library's constraints.

.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::

      * - **Function**

      * - | ``float j0f(float x)``
          | Returns the value of the Bessel function of the first kind of order 0 for :math:`x`.

      * - | ``float j1f(float x)``
          | Returns the value of the Bessel function of the first kind of order 1 for :math:`x`.

      * - | ``float jnf(int n, float x)``
          | Returns the value of the Bessel function of the first kind of order n for :math:`x`.

      * - | ``float y0f(float x)``
          | Returns the value of the Bessel function of the second kind of order 0 for :math:`x`.

      * - | ``float y1f(float x)``
          | Returns the value of the Bessel function of the second kind of order 1 for :math:`x`.

      * - | ``float ynf(int n, float x)``
          | Returns the value of the Bessel function of the second kind of order n for :math:`x`.

      * - | ``float erfcinvf(float x)``
          | Returns the inverse complementary function of :math:`x`.

      * - | ``float erfinvf(float x)``
          | Returns the inverse error function of :math:`x`.

      * - | ``float normcdff(float y)``
          | Returns the standard normal cumulative distribution function.

      * - | ``float normcdfinvf(float y)``
          | Returns the inverse of the standard normal cumulative distribution function.

  .. tab-item:: Double Precision Floating-point

    .. list-table::

      * - **Function**

      * - | ``double j0(double x)``
          | Returns the value of the Bessel function of the first kind of order 0 for :math:`x`.

      * - | ``double j1(double x)``
          | Returns the value of the Bessel function of the first kind of order 1 for :math:`x`.

      * - | ``double jn(int n, double x)``
          | Returns the value of the Bessel function of the first kind of order n for :math:`x`.

      * - | ``double y0(double x)``
          | Returns the value of the Bessel function of the second kind of order 0 for :math:`x`.

      * - | ``double y1(double x)``
          | Returns the value of the Bessel function of the second kind of order 1 for :math:`x`.

      * - | ``double yn(int n, double x)``
          | Returns the value of the Bessel function of the second kind of order n for :math:`x`.

      * - | ``double erfcinv(double x)``
          | Returns the inverse complementary function of :math:`x`.

      * - | ``double erfinv(double x)``
          | Returns the inverse error function of :math:`x`.

      * - | ``double normcdf(double y)``
          | Returns the standard normal cumulative distribution function.

      * - | ``double normcdfinv(double y)``
          | Returns the inverse of the standard normal cumulative distribution function.

Unsupported
-----------

This table lists functions that are not supported by HIP.

.. tab-set::

  .. tab-item:: Single Precision Floating-point

    .. list-table::

      * - **Function**

      * - | ``float cyl_bessel_i0f(float x)``
          | Returns the value of the regular modified cylindrical Bessel function of order 0 for :math:`x`.

      * - | ``float cyl_bessel_i1f(float x)``
          | Returns the value of the regular modified cylindrical Bessel function of order 1 for :math:`x`.

  .. tab-item:: Double Precision Floating-point

    .. list-table::

      * - **Function**

      * - | ``double cyl_bessel_i0(double x)``
          | Returns the value of the regular modified cylindrical Bessel function of order 0 for :math:`x`.

      * - | ``double cyl_bessel_i1(double x)``
          | Returns the value of the regular modified cylindrical Bessel function of order 1 for :math:`x`.

Intrinsic mathematical functions
================================

Intrinsic math functions are optimized for performance on HIP-supported hardware. These functions often
trade some precision for faster execution, making them ideal for applications where computational
efficiency is a priority over strict numerical accuracy. Note that intrinsics are supported on device only.

Floating-point Intrinsics
-------------------------

.. note::

  Only the nearest-even rounding mode is supported by default on AMD GPUs. The ``_rz``, ``_ru``, and ``_rd``
  suffixed intrinsic functions exist in the HIP AMD backend if the
  ``OCML_BASIC_ROUNDED_OPERATIONS`` macro is defined.

.. list-table:: Single precision intrinsics mathematical functions
    :widths: 50,20,30

    * - **Function**
      - **Test Range**
      - **ULP Difference of Maximum Absolute Error**

    * - | ``float __cosf(float x)``
        | Returns the fast approximate cosine of :math:`x`.
      - :math:`x \in [-\pi, \pi]`
      - 4

    * - | ``float __exp10f(float x)``
        | Returns the fast approximate for 10 :sup:`x`.
      - :math:`x \in [-4, 4]`
      - 18

    * - | ``float __expf(float x)``
        | Returns the fast approximate for e :sup:`x`.
      - :math:`x \in [-10, 10]`
      - 6

    * - | ``float __fadd_rn(float x, float y)``
        | Add two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-1000, 1000]`
        | :math:`y \in [-1000, 1000]`
      - 0

    * - | ``float __fdiv_rn(float x, float y)``
        | Divide two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-100, 100]`
        | :math:`y \in [-100, 100]`
      - 0

    * - | ``float __fmaf_rn(float x, float y, float z)``
        | Returns ``x × y + z`` as a single operation in round-to-nearest-even mode.
      - | :math:`x \in [-100, 100]`
        | :math:`y \in [-10, 10]`
        | :math:`z \in [-10, 10]`
      - 0

    * - | ``float __fmul_rn(float x, float y)``
        | Multiply two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-100, 100]`
        | :math:`y \in [-100, 100]`
      - 0

    * - | ``float __frcp_rn(float x)``
        | Returns ``1 / x`` in round-to-nearest-even mode.
      - :math:`x \in [-100, 100]`
      - 0

    * - | ``float __frsqrt_rn(float x)``
        | Returns ``1 / √x`` in round-to-nearest-even mode.
      - :math:`x \in [0.01, 100]`
      - 1

    * - | ``float __fsqrt_rn(float x)``
        | Returns ``√x`` in round-to-nearest-even mode.
      - :math:`x \in [0, 100]`
      - 1

    * - | ``float __fsub_rn(float x, float y)``
        | Subtract two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-1000, 1000]`
        | :math:`y \in [-1000, 1000]`
      - 0

    * - | ``float __log10f(float x)``
        | Returns the fast approximate for base 10 logarithm of :math:`x`.
      - :math:`x \in [10^{-6}, 10^6]`
      - 2

    * - | ``float __log2f(float x)``
        | Returns the fast approximate for base 2 logarithm of :math:`x`.
      - :math:`x \in [10^{-6}, 10^6]`
      - 1

    * - | ``float __logf(float x)``
        | Returns the fast approximate for natural logarithm of :math:`x`.
      - :math:`x \in [10^{-6}, 10^6]`
      - 2

    * - | ``float __powf(float x, float y)``
        | Returns the fast approximate of x :sup:`y`.
      - | :math:`x \in [-4, 4]`
        | :math:`y \in [-2, 2]`
      - 1

    * - | ``float __saturatef(float x)``
        | Clamp :math:`x` to [+0.0, 1.0].
      - :math:`x \in [-2, 3]`
      - 0

    * - | ``float __sincosf(float x, float* sinptr, float* cosptr)``
        | Returns the fast approximate of sine and cosine of :math:`x`.
      - :math:`x \in [-3, 3]`
      - | ``sin``: 18
        | ``cos``: 4

    * - | ``float __sinf(float x)``
        | Returns the fast approximate sine of :math:`x`.
      - :math:`x \in [-\pi, \pi]`
      - 18

    * - | ``float __tanf(float x)``
        | Returns the fast approximate tangent of :math:`x`.
      - :math:`x \in [-1.47\pi, 1.47\pi]`
      - 1

.. list-table:: Double precision intrinsics mathematical functions
    :widths: 50,20,30

    * - **Function**
      - **Test Range**
      - **ULP Difference of Maximum Absolute Error**

    * - | ``double __dadd_rn(double x, double y)``
        | Add two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-1000, 1000]`
        | :math:`y \in [-1000, 1000]`
      - 0

    * - | ``double __ddiv_rn(double x, double y)``
        | Divide two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-100, 100]`
        | :math:`y \in [-100, 100]`
      - 0

    * - | ``double __dmul_rn(double x, double y)``
        | Multiply two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-100, 100]`
        | :math:`y \in [-100, 100]`
      - 0

    * - | ``double __drcp_rn(double x)``
        | Returns ``1 / x`` in round-to-nearest-even mode.
      - :math:`x \in [-100, 100]`
      - 0

    * - | ``double __dsqrt_rn(double x)``
        | Returns ``√x`` in round-to-nearest-even mode.
      - :math:`x \in [0, 100]`
      - 0

    * - | ``double __dsub_rn(double x, double y)``
        | Subtract two floating-point values in round-to-nearest-even mode.
      - | :math:`x \in [-1000, 1000]`
        | :math:`y \in [-1000, 1000]`
      - 0

    * - | ``double __fma_rn(double x, double y, double z)``
        | Returns ``x × y + z`` as a single operation in round-to-nearest-even mode.
      - | :math:`x \in [-100, 100]`
        | :math:`y \in [-10, 10]`
        | :math:`z \in [-10, 10]`
      - 0

Integer intrinsics
------------------

This section covers HIP integer intrinsic functions. ULP error values are omitted
since they only apply to floating-point operations, not integer arithmetic.

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

    * - | ``unsigned int __ffs(int x)`` [1]_
        | Returns the position of the first set bit in a 32 bit integer.
        | Note: if ``x`` is ``0``, will return ``0``

    * - | ``unsigned int __ffsll(long long int x)`` [1]_
        | Returns the position of the first set bit in a 64 bit signed integer.
        | Note: if ``x`` is ``0``, will return ``0``

    * - | ``unsigned int __fns32(unsigned int mask, unsigned int base, int offset)``
        | Find the position of the n-th set to 1 bit in a 32-bit integer.
        | Note: this intrinsic is emulated via software, so performance can be potentially slower

    * - | ``unsigned int __fns64(unsigned long long int mask, unsigned int base, int offset)``
        | Find the position of the n-th set to 1 bit in a 64-bit integer.
        | Note: this intrinsic is emulated via software, so performance can be potentially slower

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

.. [1] The HIP-Clang implementation of ``__ffs()`` and ``__ffsll()`` contains code to add a constant +1 to produce the ``ffs`` result format.
       For the cases where this overhead is not acceptable and programmer is willing to specialize for the platform,
       HIP-Clang provides ``__lastbit_u32_u32(unsigned int input)`` and ``__lastbit_u32_u64(unsigned long long int input)``.
       The index returned by ``__lastbit_`` instructions starts at -1, while for ``ffs`` the index starts at 0.
