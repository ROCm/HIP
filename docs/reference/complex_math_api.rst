.. meta::
  :description: This chapter describes the complex math functions that are accessible in HIP.
  :keywords: AMD, ROCm, HIP, CUDA, complex math functions, HIP complex math functions

.. _complex_math_api_reference:

********************************************************************************
HIP complex math API
********************************************************************************

HIP provides built-in support for complex number operations through specialized types and functions,
available for both single-precision (float) and double-precision (double) calculations. All complex types
and functions are available on both host and device.

For any complex number ``z``, the form is:

.. math::

   z = x + yi

where ``x`` is the real part and ``y`` is the imaginary part.

Complex Number Types
====================

A brief overview of the specialized data types used to represent complex numbers in HIP, available
in both single and double precision formats.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - Type
      - Description

    * - ``hipFloatComplex``
      - | Complex number using single-precision (float) values
        | (note: ``hipComplex`` is an alias of ``hipFloatComplex``)

    * - ``hipDoubleComplex``
      - Complex number using double-precision (double) values

Complex Number Functions
========================

.. note::

  Changes have been made to small vector constructors for ``hipComplex`` and ``hipFloatComplex``
  initialization, such as ``float2`` and ``int4``. If your code previously relied
  on a single value to initialize all components within a vector or complex type, you might need
  to update your code.

A comprehensive collection of functions for creating and manipulating complex numbers, organized by
functional categories for easy reference.

Type Construction
-----------------

Functions for creating complex number objects and extracting their real and imaginary components.

.. tab-set::

   .. tab-item:: Single Precision

      .. list-table::
         :header-rows: 1
         :widths: 40 60

         * - Function
           - Description

         * - | ``hipFloatComplex``
             | ``make_hipFloatComplex(``
             |    ``float a,``
             |    ``float b``
             | ``)``
           - | Creates a complex number
             | (note: ``make_hipComplex`` is an alias of ``make_hipFloatComplex``)
             | :math:`z = a + bi`

         * - | ``float``
             | ``hipCrealf(``
             |    ``hipFloatComplex z``
             | ``)``
           - | Returns real part of z
             | :math:`\Re(z) = x`

         * - | ``float``
             | ``hipCimagf(``
             |    ``hipFloatComplex z``
             | ``)``
           - | Returns imaginary part of z
             | :math:`\Im(z) = y`

   .. tab-item:: Double Precision

      .. list-table::
         :header-rows: 1
         :widths: 40 60

         * - Function
           - Description

         * - | ``hipDoubleComplex``
             | ``make_hipDoubleComplex(``
             |    ``double a,``
             |    ``double b``
             | ``)``
           - | Creates a complex number
             | :math:`z = a + bi`

         * - | ``double``
             | ``hipCreal(``
             |    ``hipDoubleComplex z``
             | ``)``
           - | Returns real part of z
             | :math:`\Re(z) = x`

         * - | ``double``
             | ``hipCimag(``
             |    ``hipDoubleComplex z``
             | ``)``
           - | Returns imaginary part of z
             | :math:`\Im(z) = y`

Basic Arithmetic
----------------

Operations for performing standard arithmetic with complex numbers, including addition,
subtraction, multiplication, division, and fused multiply-add.

.. tab-set::

   .. tab-item:: Single Precision

      .. list-table::
         :header-rows: 1
         :widths: 40 60

         * - Function
           - Description

         * - | ``hipFloatComplex``
             | ``hipCaddf(``
             |    ``hipFloatComplex p,``
             |    ``hipFloatComplex q``
             | ``)``
           - | Addition of two single-precision complex values
             | :math:`(a + bi) + (c + di) = (a + c) + (b + d)i`

         * - | ``hipFloatComplex``
             | ``hipCsubf(``
             |    ``hipFloatComplex p,``
             |    ``hipFloatComplex q``
             | ``)``
           - | Subtraction of two single-precision complex values
             | :math:`(a + bi) - (c + di) = (a - c) + (b - d)i`

         * - | ``hipFloatComplex``
             | ``hipCmulf(``
             |    ``hipFloatComplex p,``
             |    ``hipFloatComplex q``
             | ``)``
           - | Multiplication of two single-precision complex values
             | :math:`(a + bi)(c + di) = (ac - bd) + (bc + ad)i`

         * - | ``hipFloatComplex``
             | ``hipCdivf(``
             |    ``hipFloatComplex p,``
             |    ``hipFloatComplex q``
             | ``)``
           - | Division of two single-precision complex values
             | :math:`\frac{a + bi}{c + di} = \frac{(ac + bd) + (bc - ad)i}{c^2 + d^2}`

         * - | ``hipFloatComplex``
             | ``hipCfmaf(``
             |    ``hipComplex p,``
             |    ``hipComplex q,``
             |    ``hipComplex r``
             | ``)``
           - | Fused multiply-add of three single-precision complex values
             | :math:`(a + bi)(c + di) + (e + fi)`

   .. tab-item:: Double Precision

      .. list-table::
         :header-rows: 1
         :widths: 40 60

         * - Function
           - Description

         * - | ``hipDoubleComplex``
             | ``hipCadd(``
             |    ``hipDoubleComplex p,``
             |    ``hipDoubleComplex q``
             | ``)``
           - | Addition of two double-precision complex values
             | :math:`(a + bi) + (c + di) = (a + c) + (b + d)i`

         * - | ``hipDoubleComplex``
             | ``hipCsub(``
             |    ``hipDoubleComplex p,``
             |    ``hipDoubleComplex q``
             | ``)``
           - | Subtraction of two double-precision complex values
             | :math:`(a + bi) - (c + di) = (a - c) + (b - d)i`

         * - | ``hipDoubleComplex``
             | ``hipCmul(``
             |    ``hipDoubleComplex p,``
             |    ``hipDoubleComplex q``
             | ``)``
           - | Multiplication of two double-precision complex values
             | :math:`(a + bi)(c + di) = (ac - bd) + (bc + ad)i`

         * - | ``hipDoubleComplex``
             | ``hipCdiv(``
             |    ``hipDoubleComplex p,``
             |    ``hipDoubleComplex q``
             | ``)``
           - | Division of two double-precision complex values
             | :math:`\frac{a + bi}{c + di} = \frac{(ac + bd) + (bc - ad)i}{c^2 + d^2}`

         * - | ``hipDoubleComplex``
             | ``hipCfma(``
             |    ``hipDoubleComplex p,``
             |    ``hipDoubleComplex q,``
             |    ``hipDoubleComplex r``
             | ``)``
           - | Fused multiply-add of three double-precision complex values
             | :math:`(a + bi)(c + di) + (e + fi)`

Complex Operations
------------------

Functions for complex-specific calculations, including conjugate determination and magnitude
(absolute value) computation.

.. tab-set::

   .. tab-item:: Single Precision

      .. list-table::
         :header-rows: 1
         :widths: 40 60

         * - Function
           - Description

         * - | ``hipFloatComplex``
             | ``hipConjf(``
             |    ``hipFloatComplex z``
             | ``)``
           - | Complex conjugate
             | :math:`\overline{a + bi} = a - bi`

         * - | ``float``
             | ``hipCabsf(``
             |    ``hipFloatComplex z``
             | ``)``
           - | Absolute value (magnitude)
             | :math:`|a + bi| = \sqrt{a^2 + b^2}`

         * - | ``float``
             | ``hipCsqabsf(``
             |    ``hipFloatComplex z``
             | ``)``
           - | Squared absolute value
             | :math:`|a + bi|^2 = a^2 + b^2`

   .. tab-item:: Double Precision

      .. list-table::
         :header-rows: 1
         :widths: 40 60

         * - Function
           - Description

         * - | ``hipDoubleComplex``
             | ``hipConj(``
             |    ``hipDoubleComplex z``
             | ``)``
           - | Complex conjugate
             | :math:`\overline{a + bi} = a - bi`

         * - | ``double``
             | ``hipCabs(``
             |    ``hipDoubleComplex z``
             | ``)``
           - | Absolute value (magnitude)
             | :math:`|a + bi| = \sqrt{a^2 + b^2}`

         * - | ``double``
             | ``hipCsqabs(``
             |    ``hipDoubleComplex z``
             | ``)``
           - | Squared absolute value
             | :math:`|a + bi|^2 = a^2 + b^2`

Type Conversion
---------------

Utility functions for conversion between single-precision and double-precision complex number formats.

.. list-table::
  :header-rows: 1
  :widths: 40 60

  * - Function
    - Description

  * - | ``hipFloatComplex``
      | ``hipComplexDoubleToFloat(``
      |    ``hipDoubleComplex z``
      | ``)``
    - Converts double-precision to single-precision complex

  * - | ``hipDoubleComplex``
      | ``hipComplexFloatToDouble(``
      |    ``hipFloatComplex z``
      | ``)``
    - Converts single-precision to double-precision complex

Example Usage
=============

The following example demonstrates using complex numbers to compute the Discrete Fourier Transform (DFT)
of a simple signal on the GPU. The DFT converts a signal from the time domain to the frequency domain.
The kernel function ``computeDFT`` shows various HIP complex math operations in action:

* Creating complex numbers with ``make_hipFloatComplex``
* Performing complex multiplication with ``hipCmulf``
* Accumulating complex values with ``hipCaddf``

The example also demonstrates proper use of complex number handling on both host and device, including
memory allocation, transfer, and validation of results between CPU and GPU implementations.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <hip/hip_complex.h>
    #include <iostream>
    #include <vector>
    #include <cmath>

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

    // Kernel to compute DFT
    __global__ void computeDFT(const float* input,
                            hipFloatComplex* output,
                            const int N)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= N) return;

        hipFloatComplex sum = make_hipFloatComplex(0.0f, 0.0f);

        for (int n = 0; n < N; n++) {
            float angle = -2.0f * M_PI * k * n / N;
            hipFloatComplex w = make_hipFloatComplex(cosf(angle), sinf(angle));
            hipFloatComplex x = make_hipFloatComplex(input[n], 0.0f);
            sum = hipCaddf(sum, hipCmulf(x, w));
        }

        output[k] = sum;
    }

    // CPU implementation of DFT for verification
    std::vector<hipFloatComplex> cpuDFT(const std::vector<float>& input) {
        const int N = input.size();
        std::vector<hipFloatComplex> result(N);

        for (int k = 0; k < N; k++) {
            hipFloatComplex sum = make_hipFloatComplex(0.0f, 0.0f);
            for (int n = 0; n < N; n++) {
                float angle = -2.0f * M_PI * k * n / N;
                hipFloatComplex w = make_hipFloatComplex(cosf(angle), sinf(angle));
                hipFloatComplex x = make_hipFloatComplex(input[n], 0.0f);
                sum = hipCaddf(sum, hipCmulf(x, w));
            }
            result[k] = sum;
        }
        return result;
    }

    int main() {
        const int N = 256;  // Signal length
        const int blockSize = 256;

        // Generate input signal: sum of two sine waves
        std::vector<float> signal(N);
        for (int i = 0; i < N; i++) {
            float t = static_cast<float>(i) / N;
            signal[i] = sinf(2.0f * M_PI * 10.0f * t) +  // 10 Hz component
                    0.5f * sinf(2.0f * M_PI * 20.0f * t);  // 20 Hz component
        }

        // Compute reference solution on CPU
        std::vector<hipFloatComplex> cpu_output = cpuDFT(signal);

        // Allocate device memory
        float* d_signal;
        hipFloatComplex* d_output;
        HIP_CHECK(hipMalloc(&d_signal, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, N * sizeof(hipFloatComplex)));

        // Copy input to device
        HIP_CHECK(hipMemcpy(d_signal, signal.data(), N * sizeof(float),
                        hipMemcpyHostToDevice));

        // Launch kernel
        dim3 grid((N + blockSize - 1) / blockSize);
        dim3 block(blockSize);
        computeDFT<<<grid, block>>>(d_signal, d_output, N);
        HIP_CHECK(hipGetLastError());

        // Get GPU results
        std::vector<hipFloatComplex> gpu_output(N);
        HIP_CHECK(hipMemcpy(gpu_output.data(), d_output, N * sizeof(hipFloatComplex),
                        hipMemcpyDeviceToHost));

        // Verify results
        bool passed = true;
        const float tolerance = 1e-5f;  // Adjust based on precision requirements

        for (int i = 0; i < N; i++) {
            float diff_real = std::abs(hipCrealf(gpu_output[i]) - hipCrealf(cpu_output[i]));
            float diff_imag = std::abs(hipCimagf(gpu_output[i]) - hipCimagf(cpu_output[i]));

            if (diff_real > tolerance || diff_imag > tolerance) {
                passed = false;
                break;
            }
        }

        std::cout << "DFT Verification: " << (passed ? "PASSED" : "FAILED") << "\n";

        // Cleanup
        HIP_CHECK(hipFree(d_signal));
        HIP_CHECK(hipFree(d_output));
        return passed ? 0 : 1;
    }
