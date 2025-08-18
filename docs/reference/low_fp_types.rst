.. meta::
    :description: This page describes the FP8 and FP16 types present in HIP.
    :keywords: AMD, ROCm, HIP, fp8, fnuz, ocp

*******************************************************************************
Low precision floating point types
*******************************************************************************

Modern computing tasks often require balancing numerical precision against hardware resources
and processing speed. Low precision floating point number formats in HIP include FP8 (Quarter Precision)
and FP16 (Half Precision), which reduce memory and bandwidth requirements compared to traditional
32-bit or 64-bit formats. The following sections detail their specifications, variants, and provide
practical guidance for implementation in HIP.

FP8 (Quarter Precision)
=======================

`FP8 (Floating Point 8-bit) numbers <https://arxiv.org/pdf/2209.05433>`_ were introduced
as a compact numerical format specifically tailored for deep learning inference. By reducing
precision while maintaining computational effectiveness, FP8 allows for significant memory
savings and improved processing speed. This makes it particularly beneficial for deploying
large-scale models with strict efficiency constraints.

Unlike traditional floating-point formats such as FP32 or even FP16, FP8 further optimizes
performance by enabling a higher volume of matrix operations per second. Its reduced bit-width
minimizes bandwidth requirements, making it an attractive choice for hardware accelerators
in deep learning applications.

There are two primary FP8 formats:

- **E4M3 Format**

  - Sign: 1 bit
  - Exponent: 4 bits
  - Mantissa: 3 bits

- **E5M2 Format**

  - Sign: 1 bit
  - Exponent: 5 bits
  - Mantissa: 2 bits

The E4M3 format offers higher precision with a narrower range, while the E5M2 format provides
a wider range at the cost of some precision.

Additionally, FP8 numbers have two representations:

- **FP8-OCP (Open Compute Project)**

  - `This <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>`_
    is a standardized format developed by the Open Compute Project to ensure compatibility
    across various hardware and software implementations.

- **FP8-FNUZ (Finite and NaN Only)**

  - A specialized format optimized for specific computations, supporting only finite and NaN values
    (no Inf support).
  - This provides one extra value of exponent and adds to the range of supported FP8 numbers.
  - **NaN Definition**: When the sign bit is set, and all other exponent and mantissa bits are zero.

The FNUZ representation provides an extra exponent value, expanding the range of representable
numbers compared to standard FP8 formats.


HIP Header
----------

The `HIP FP8 header <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_fp8.h>`_
defines the FP8 ocp/fnuz numbers.

Supported Devices
-----------------

Different GPU models support different FP8 formats. Here's a breakdown:

.. list-table:: Supported devices for fp8 numbers
    :header-rows: 1

    * - Device Type
      - FNUZ FP8
      - OCP FP8
    * - Host
      - Yes
      - Yes
    * - CDNA1
      - No
      - No
    * - CDNA2
      - No
      - No
    * - CDNA3
      - Yes
      - No
    * - RDNA2
      - No
      - No
    * - RDNA3
      - No
      - No

Using FP8 Numbers in HIP Programs
---------------------------------

To use the FP8 numbers inside HIP programs.

.. code-block:: cpp

    #include <hip/hip_fp8.h>

FP8 numbers can be used on CPU side:

.. code-block:: cpp

    __hip_fp8_storage_t convert_float_to_fp8(
      float in, /* Input val */
      __hip_fp8_interpretation_t interpret, /* interpretation of number E4M3/E5M2 */
      __hip_saturation_t sat /* Saturation behavior */
      ) {
      return __hip_cvt_float_to_fp8(in, sat, interpret);
    }

The same can be done in kernels as well.

.. code-block:: cpp

    __device__ __hip_fp8_storage_t d_convert_float_to_fp8(
      float in,
      __hip_fp8_interpretation_t interpret,
      __hip_saturation_t sat) {
      return __hip_cvt_float_to_fp8(in, sat, interpret);
    }

Note: On a gfx94x GPU, the type will default to the fnuz type.

The following code example does roundtrip FP8 conversions on both the CPU and GPU and compares the results.

.. code-block:: cpp

      #include <hip/hip_fp8.h>
      #include <hip/hip_runtime.h>
      #include <iostream>
      #include <vector>

      #define hip_check(hip_call)                                                    \
      {                                                                              \
          auto hip_res = hip_call;                                                   \
          if (hip_res != hipSuccess) {                                               \
            std::cerr << "Failed in HIP call: " << #hip_call \
                      << " at " << __FILE__ << ":" << __LINE__ \
                      << " with error: " << hipGetErrorString(hip_res) << std::endl; \
            std::abort();                                                            \
          }                                                                          \
      }

      __device__ __hip_fp8_storage_t d_convert_float_to_fp8(
          float in, __hip_fp8_interpretation_t interpret, __hip_saturation_t sat) {
          return __hip_cvt_float_to_fp8(in, sat, interpret);
      }

      __device__ float d_convert_fp8_to_float(float in,
                                              __hip_fp8_interpretation_t interpret) {
          __half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
          return hf;
      }

      __global__ void float_to_fp8_to_float(float *in,
                                          __hip_fp8_interpretation_t interpret,
                                          __hip_saturation_t sat, float *out,
                                          size_t size) {
          int i = threadIdx.x;
          if (i < size) {
              auto fp8 = d_convert_float_to_fp8(in[i], interpret, sat);
              out[i] = d_convert_fp8_to_float(fp8, interpret);
          }
      }

      __hip_fp8_storage_t
      convert_float_to_fp8(float in, /* Input val */
                          __hip_fp8_interpretation_t
                              interpret, /* interpretation of number E4M3/E5M2 */
                          __hip_saturation_t sat /* Saturation behavior */
      ) {
          return __hip_cvt_float_to_fp8(in, sat, interpret);
      }

      float convert_fp8_to_float(
          __hip_fp8_storage_t in, /* Input val */
          __hip_fp8_interpretation_t
              interpret /* interpretation of number E4M3/E5M2 */
      ) {
          __half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
          return hf;
      }

      int main() {
          constexpr size_t size = 32;
          hipDeviceProp_t prop;
          hip_check(hipGetDeviceProperties(&prop, 0));
          bool is_supported = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos); // gfx94x
          if(!is_supported) {
              std::cerr << "Need a gfx94x, but found: " << prop.gcnArchName << std::endl;
              std::cerr << "No device conversions are supported, only host conversions are supported." << std::endl;
              return -1;
          }

          const __hip_fp8_interpretation_t interpret = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos)
                                                          ? __HIP_E4M3_FNUZ // gfx94x
                                                          : __HIP_E4M3;
          constexpr __hip_saturation_t sat = __HIP_SATFINITE;

          std::vector<float> in;
          in.reserve(size);
          for (size_t i = 0; i < size; i++) {
              in.push_back(i + 1.1f);
          }

          std::cout << "Converting float to fp8 and back..." << std::endl;
          // CPU convert
          std::vector<float> cpu_out;
          cpu_out.reserve(size);
          for (const auto &fval : in) {
              auto fp8 = convert_float_to_fp8(fval, interpret, sat);
              cpu_out.push_back(convert_fp8_to_float(fp8, interpret));
          }

          // GPU convert
          float *d_in, *d_out;
          hip_check(hipMalloc(&d_in, sizeof(float) * size));
          hip_check(hipMalloc(&d_out, sizeof(float) * size));

          hip_check(hipMemcpy(d_in, in.data(), sizeof(float) * in.size(),
                              hipMemcpyHostToDevice));

          float_to_fp8_to_float<<<1, size>>>(d_in, interpret, sat, d_out, size);

          std::vector<float> gpu_out(size, 0.0f);
          hip_check(hipMemcpy(gpu_out.data(), d_out, sizeof(float) * gpu_out.size(),
                              hipMemcpyDeviceToHost));

          hip_check(hipFree(d_in));
          hip_check(hipFree(d_out));

          // Validation
          for (size_t i = 0; i < size; i++) {
              if (cpu_out[i] != gpu_out[i]) {
                  std::cerr << "cpu round trip result: " << cpu_out[i]
                            << " - gpu round trip result: " << gpu_out[i] << std::endl;
                  std::abort();
              }
          }
          std::cout << "...CPU and GPU round trip convert matches." << std::endl;
      }

There are C++ style classes available as well.

.. code-block:: cpp

    __hip_fp8_e4m3_fnuz fp8_val(1.1f); // gfx94x
    __hip_fp8_e4m3 fp8_val(1.1f);

Each type of FP8 number has its own class:

- __hip_fp8_e4m3
- __hip_fp8_e5m2
- __hip_fp8_e4m3_fnuz
- __hip_fp8_e5m2_fnuz

There is support of vector of FP8 types.

- __hip_fp8x2_e4m3:      holds 2 values of OCP FP8 e4m3 numbers
- __hip_fp8x4_e4m3:      holds 4 values of OCP FP8 e4m3 numbers
- __hip_fp8x2_e5m2:      holds 2 values of OCP FP8 e5m2 numbers
- __hip_fp8x4_e5m2:      holds 4 values of OCP FP8 e5m2 numbers
- __hip_fp8x2_e4m3_fnuz: holds 2 values of FP8 fnuz e4m3 numbers
- __hip_fp8x4_e4m3_fnuz: holds 4 values of FP8 fnuz e4m3 numbers
- __hip_fp8x2_e5m2_fnuz: holds 2 values of FP8 fnuz e5m2 numbers
- __hip_fp8x4_e5m2_fnuz: holds 4 values of FP8 fnuz e5m2 numbers

FNUZ extensions will be available on gfx94x only.

FP16 (Half Precision)
=====================

FP16 (Floating Point 16-bit) numbers offer a balance between precision and
efficiency, making them a widely adopted standard for accelerating deep learning
inference. With higher precision than FP8 but lower memory requirements than FP32,
FP16 enables faster computations while preserving model accuracy.

Deep learning workloads often involve massive datasets and complex calculations,
making FP32 computationally expensive. FP16 helps mitigate these costs by reducing
storage and bandwidth demands, allowing for increased throughput without significant
loss of numerical stability. This format is particularly useful for training and
inference in GPUs and TPUs optimized for half-precision arithmetic.

There are two primary FP16 formats:

- **float16 Format**

  - Sign: 1 bit
  - Exponent: 5 bits
  - Mantissa: 10 bits

- **bfloat16 Format**

  - Sign: 1 bit
  - Exponent: 8 bits
  - Mantissa: 7 bits

The float16 format offers higher precision with a narrower range, while the bfloat16
format provides a wider range at the cost of some precision.

Additionally, FP16 numbers have standardized representations developed by industry
initiatives to ensure compatibility across various hardware and software implementations.
Unlike FP8, which has specific representations like OCP and FNUZ, FP16 is more uniformly
supported with its two main formats, float16 and bfloat16.

HIP Header
----------

The `HIP FP16 header <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_fp16.h>`_
defines the float16 format.

The `HIP BF16 header <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_bf16.h>`_
defines the bfloat16 format.

Supported Devices
-----------------

Different GPU models support different FP16 formats. Here's a breakdown:

.. list-table:: Supported devices for fp16 numbers
    :header-rows: 1

    * - Device Type
      - float16
      - bfloat16
    * - Host
      - Yes
      - Yes
    * - CDNA1
      - Yes
      - Yes
    * - CDNA2
      - Yes
      - Yes
    * - CDNA3
      - Yes
      - Yes
    * - RDNA2
      - Yes
      - Yes
    * - RDNA3
      - Yes
      - Yes

Using FP16 Numbers in HIP Programs
----------------------------------

To use the FP16 numbers inside HIP programs.

.. code-block:: cpp

    #include <hip/hip_fp16.h> // for float16
    #include <hip/hip_bf16.h> // for bfloat16

The following code example adds two float16 values on the GPU and compares the results
against summed float values on the CPU.

.. code-block:: cpp

      #include <hip/hip_fp16.h>
      #include <hip/hip_runtime.h>
      #include <iostream>
      #include <vector>

      #define hip_check(hip_call)                                                    \
      {                                                                              \
          auto hip_res = hip_call;                                                   \
          if (hip_res != hipSuccess) {                                               \
              std::cerr << "Failed in HIP call: " << #hip_call \
                        << " at " << __FILE__ << ":" << __LINE__ \
                        << " with error: " << hipGetErrorString(hip_res) << std::endl; \
              std::abort();                                                            \
          }                                                                          \
      }

      __global__ void add_half_precision(__half* in1, __half* in2, float* out, size_t size) {
          int idx = threadIdx.x;
          if (idx < size) {
              // Load as half, perform addition in float, store as float
              float sum = __half2float(in1[idx] + in2[idx]);
              out[idx] = sum;
          }
      }

      int main() {
          constexpr size_t size = 32;
          constexpr float tolerance = 1e-1f;  // Allowable numerical difference

          // Initialize input vectors as floats
          std::vector<float> in1(size), in2(size);
          for (size_t i = 0; i < size; i++) {
              in1[i] = i + 1.1f;
              in2[i] = i + 2.2f;
          }

          // Compute expected results in full precision on CPU
          std::vector<float> cpu_out(size);
          for (size_t i = 0; i < size; i++) {
              cpu_out[i] = in1[i] + in2[i];  // Direct float addition
          }

          // Allocate device memory (store input as half, output as float)
          __half *d_in1, *d_in2;
          float *d_out;
          hip_check(hipMalloc(&d_in1, sizeof(__half) * size));
          hip_check(hipMalloc(&d_in2, sizeof(__half) * size));
          hip_check(hipMalloc(&d_out, sizeof(float) * size));

          // Convert input to half and copy to device
          std::vector<__half> in1_half(size), in2_half(size);
          for (size_t i = 0; i < size; i++) {
              in1_half[i] = __float2half(in1[i]);
              in2_half[i] = __float2half(in2[i]);
          }

          hip_check(hipMemcpy(d_in1, in1_half.data(), sizeof(__half) * size, hipMemcpyHostToDevice));
          hip_check(hipMemcpy(d_in2, in2_half.data(), sizeof(__half) * size, hipMemcpyHostToDevice));

          // Launch kernel
          add_half_precision<<<1, size>>>(d_in1, d_in2, d_out, size);

          // Copy result back to host
          std::vector<float> gpu_out(size, 0.0f);
          hip_check(hipMemcpy(gpu_out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));

          // Free device memory
          hip_check(hipFree(d_in1));
          hip_check(hipFree(d_in2));
          hip_check(hipFree(d_out));

          // Validation with tolerance
          for (size_t i = 0; i < size; i++) {
              if (std::fabs(cpu_out[i] - gpu_out[i]) > tolerance) {
                  std::cerr << "Mismatch at index " << i << ": CPU result = " << cpu_out[i]
                            << ", GPU result = " << gpu_out[i] << std::endl;
                  std::abort();
              }
          }

          std::cout << "Success: CPU and GPU half-precision addition match within tolerance!" << std::endl;
      }


There are C++ style classes available as well.

.. code-block:: cpp

    __half fp16_val(1.1f);           // float16
    __hip_bfloat16 fp16_val(1.1f);   // bfloat16

Each type of FP16 number has its own class:

- __half
- __hip_bfloat16

There is support of vector of FP16 types.

- __half2:              holds 2 values of float16 numbers
- __hip_bfloat162:      holds 2 values of bfloat16 numbers
