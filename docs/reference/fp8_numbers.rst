.. meta::
    :description: This page describes FP4, FP6 and FP8 numbers present in HIP.
    :keywords: AMD, ROCm, HIP, fp8, fnuz, ocp, microscaling, mx

FP8 Numbers
===========

`FP8 numbers <https://arxiv.org/pdf/2209.05433>`_ were introduced to accelerate deep learning inferencing. They provide higher throughput of matrix operations because the smaller size allows more of them in the available fixed memory.

HIP has two FP8 number representations called *FP8-OCP* and *FP8-FNUZ*.

Open Compute Project(OCP) number definition can be found `here <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>`_.

Definition of FNUZ: fnuz suffix means only finite and NaN values are supported. Unlike other types, Inf are not supported.
NaN is when sign bit is set and all other exponent and mantissa bits are 0. All other values are finite.
This provides one extra value of exponent and adds to the range of supported FP8 numbers.

Definitions
===========

FP8
===

FP8 numbers are composed of a sign, an exponent and a mantissa. Their sizes are dependent on the format.
There are two formats of FP8 numbers, E4M3 and E5M2.

- E4M3: 1 bit sign, 4 bit exponent, 3 bit mantissa
- E5M2: 1 bit sign, 5 bit exponent, 2 bit mantissa

FP6
===

There are two formats of FP6 numbers, E2M3 and E3M2.

- E2M3: 1 bit sign, 2 bit exponent, 3 bit mantissa
- E3M2: 1 bit sign, 3 bit exponent, 2 bit mantissa

FP4
===

There is one format for FP4, E2M1.

- E2M1: 1 bit sign, 2 bit exponent, 1 bit mantissa

HIP Header
==========

The `HIP header <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_fp8.h>`_ defines the FP8 ocp/fnuz numbers.

Supported Devices for FP8 Types
===============================

.. list-table:: Supported devices for fp8 numbers
    :header-rows: 1

    * - Device Type
      - FNUZ FP8
      - OCP FP8
    * - Host
      - Yes
      - Yes
    * - gfx942
      - Yes
      - No
    * - gfx1200/gfx1201
      - No
      - Yes

Usage
=====

To use the FP8 numbers inside HIP programs.

.. code-block:: c

  #include <hip/hip_fp8.h>

FP8 numbers can be used on CPU side:

.. code-block:: c

  __hip_fp8_storage_t convert_float_to_fp8(
    float in, /* Input val */
    __hip_fp8_interpretation_t interpret, /* interpretation of number E4M3/E5M2 */
    __hip_saturation_t sat /* Saturation behavior */
    ) {
    return __hip_cvt_float_to_fp8(in, sat, interpret);
  }

The same can be done in kernels as well.

.. code-block:: c

  __device__ __hip_fp8_storage_t d_convert_float_to_fp8(
     float in,
    __hip_fp8_interpretation_t interpret,
    __hip_saturation_t sat) {
    return __hip_cvt_float_to_fp8(in, sat, interpret);
  }

An important thing to note here is if you use this on gfx94x GPU, it will be fnuz number but on any other GPU it will be an OCP number.

The following code example does roundtrip FP8 conversions on both the CPU and GPU and compares the results.

.. code-block:: c

    #include <hip/hip_fp8.h>
    #include <hip/hip_runtime.h>
    #include <iostream>
    #include <vector>

    #define hip_check(hip_call)                                                    \
    {                                                                              \
        auto hip_res = hip_call;                                                   \
        if (hip_res != hipSuccess) {                                               \
          std::cerr << "Failed in hip call: " << #hip_call                         \
                    << " with error: " << hipGetErrorName(hip_res) << std::endl;   \
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
        bool is_supported = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos) || // gfx94x
                            (std::string(prop.gcnArchName).find("gfx120") != std::string::npos);  // gfx120x
        if(!is_supported) {
            std::cerr << "Need a gfx94x or gfx120x, but found: " << prop.gcnArchName << std::endl;
            std::cerr << "No device conversions are supported, only host conversions are supported." << std::endl;
            return -1;
        }

        const __hip_fp8_interpretation_t interpret = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos)
                                                        ? __HIP_E4M3_FNUZ // gfx94x
                                                        : __HIP_E4M3;     // gfx120x
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

.. code-block:: c

    __hip_fp8_e4m3_fnuz fp8_val(1.1f); // gfx94x
    __hip_fp8_e4m3 fp8_val(1.1f);      // gfx120x

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


FP4, FP6, FP8 microscaling formats
==================================

OCP introduced new set of microscaling data formats. `spec link <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_.

HIP supports these microscaling formats and there is hardware acceleration available for these formats.

Hardware Acceleration and Rounding Behavior
===========================================

The only GPU that will accelerate microscaling formats is `gfx950`. For other GPUs, software path will be taken.

The default rounding mode is round to nearest and does not support other rounding types. If any other rounding mode is requested, they will be ignored and the value will be rounded to nearest.

FP4 numbers
===========

FP4 has the format of E2M1.

Usage
=====

To use the fp4 functionality, you need to use `#include <hip/hip_fp4.h>`.

Example:

.. code-block:: c

  __host__ __device__ __hip_fp4_storage_t convert_float_to_fp4(
     float in,
    __hip_fp4_interpretation_t interpret,
    const hipRoundMode round) {
    return __hip_cvt_float_to_fp4(in, interpret, round);
  }

Available APIs
==============

All the APIs are available on host and device.

.. list-table:: C APIs
    :header-rows: 1

    * - API
      - Notes
    * - `__hip_fp4_storage_t  __hip_cvt_bfloat16raw_to_fp4(const __hip_bfloat16_raw, const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a __hip_bfloat16_raw to fp4.
    * - `__hip_fp4x2_storage_t __hip_cvt_bfloat16raw2_to_fp4x2(const __hip_bfloat162_raw, const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a packed __hip_bfloat162_raw to packed fp4x2.
    * - `__hip_fp4_storage_t __hip_cvt_double_to_fp4(const double, const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a double to fp4.
    * - `__hip_fp4x2_storage_t __hip_cvt_double2_to_fp4x2(const double2, const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a packed double2 to packed fp4x2.
    * - `__hip_fp4_storage_t __hip_cvt_float_to_fp4(const float, const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a float to fp4.
    * - `__hip_fp4x2_storage_t __hip_cvt_float2_to_fp4x2(const float2, const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a packed float2 to packed fp4x2.
    * - `__hip_fp4_storage_t  __hip_cvt_halfraw_to_fp4(const __half_raw , const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a __half_raw to fp4.
    * - `__hip_fp4x2_storage_t __hip_cvt_halfraw2_to_fp4x2(const __half_raw2, const __hip_fp4_interpretation_t, const enum hipRoundMode)`
      - Convert a packed __half_raw2 to packed fp4x2.
    * - `__half_raw __hip_cvt_fp4_to_halfraw(const __hip_fp4_storage_t, const __hip_fp4_interpretation_t)`
      - Convert a fp4 value to __half_raw.
    * - `__half2_raw __hip_cvt_fp4x2_to_halfraw2(const __hip_fp4x2_storage_t, const __hip_fp4_interpretation_t)`
      - Convert a packed fp4x2 value to __half2_raw.

There are C++ style classes available.

.. list-table:: C++ types
    :header-rows: 1

    * - struct
      - Notes
    * - `__hip_fp4_e2m1`
      - A single value for fp4 e2m1 type
    * - `__hip_fp4x2_e2m1`
      - A two packed value for fp4 e2m1 type
    * - `__hip_fp4x4_e2m1`
      - A four packed value for fp4 e2m1 type

For FP4 E2M1 format.

.. code-block:: c

  struct __hip_fp4_e2m1 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP4_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp4_e2m1(const __half); // half to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const __hip_bfloat16); // __hip_bfloat16 to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const double); // double to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const float); // float to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const int); // int to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const long); // long to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const long long); // long long to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const short); // short to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const unsigned); // unsigned to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const unsigned long); // unsigned long to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const unsigned long long); // unsigned long long to fp4
    __host__ __device__ explicit __hip_fp4_e2m1(const unsigned short); // unsigned short to fp4

    // Operators
    // These operators are hidden if __HIP_NO_FP4_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator __half_raw(); // return a __half_raw
    __host__ __device__ operator __hip_bfloat16_raw(); // return a __hip_bfloat16_raw
    __host__ __device__ operator float(); // return a float
    __host__ __device__ operator double(); // return a double
  };

For packed FP4x2 E2M1 format.

.. code-block:: c

  struct __hip_fp4x2_e2m1 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP4_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp4x2_e2m1(const __half2); // half2 to fp4x2
    __host__ __device__ explicit __hip_fp4x2_e2m1(const __hip_bfloat162); // __hip_bfloat162 to fp4x2
    __host__ __device__ explicit __hip_fp4x2_e2m1(const double2); // double2 to fp4x2
    __host__ __device__ explicit __hip_fp4x2_e2m1(const float2); // float2 to fp4x2

    // Operators
    // These operators are hidden if __HIP_NO_FP4_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator __half2_raw(); // return a __half2_raw
    __host__ __device__ operator __hip_bfloat162_raw(); // return a __hip_bfloat162_raw
    __host__ __device__ operator float2(); // return a float2
    __host__ __device__ operator double2(); // return a double2
  };

For packed FP4x4 E2M1 format.

.. code-block:: c

  struct __hip_fp4x4_e2m1 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP4_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp4x4_e2m1(const __half2, const __half2); // convert two half2 to fp4x4
    __host__ __device__ explicit __hip_fp4x4_e2m1(const __hip_bfloat162, const __hip_bfloat162); // convert two __hip_bfloat162 to fp4x4
    __host__ __device__ explicit __hip_fp4x4_e2m1(const double4); // double4 to fp4x4
    __host__ __device__ explicit __hip_fp4x4_e2m1(const float4); // float4 to fp4x4

    // Operators
    // These operators are hidden if __HIP_NO_FP4_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator float4(); // return a float4
    __host__ __device__ operator double4(); // return a double4
  };


FP6 numbers
===========

FP6 has two formats, E2M3 and E3M2.

Usage
=====

To use the fp4 functionality, you need to use `#include <hip/hip_fp4.h>`.

Available APIs
==============

All the APIs are available on host and device.

.. list-table:: C APIs
    :header-rows: 1

    * - API
      - Notes
    * - `__hip_fp6_storage_t __hip_cvt_bfloat16raw_to_fp6(const __hip_bfloat16_raw, const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a __hip_bfloat16_raw to fp6.
    * - `__hip_fp6x2_storage_t __hip_cvt_bfloat16raw2_to_fp6x2(const __hip_bfloat162_raw, const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a packed __hip_bfloat162_raw to packed fp6x2.
    * - `__hip_fp6_storage_t __hip_cvt_double_to_fp6(const double, const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a double to fp6.
    * - `__hip_fp6x2_storage_t __hip_cvt_double2_to_fp6x2(const double2, const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a packed double2 to packed fp6x2.
    * - `__hip_fp6_storage_t __hip_cvt_float_to_fp6(const float, const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a float to fp6.
    * - `__hip_fp6x2_storage_t __hip_cvt_float2_to_fp6x2(const float2, const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a packed float2 to packed fp6x2.
    * - `__hip_fp6_storage_t  __hip_cvt_halfraw_to_fp6(const __half_raw , const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a __half_raw to fp6.
    * - `__hip_fp6x2_storage_t __hip_cvt_halfraw2_to_fp6x2(const __half_raw2, const __hip_fp6_interpretation_t, const enum hipRoundMode)`
      - Convert a packed __half_raw2 to packed fp6x2.
    * - `__half_raw __hip_cvt_fp6_to_halfraw(const __hip_fp6_storage_t, const __hip_fp6_interpretation_t)`
      - Convert a fp6 value to __half_raw.
    * - `__half2_raw __hip_cvt_fp6x2_to_halfraw2(const __hip_fp6x2_storage_t, const __hip_fp6_interpretation_t)`
      - Convert a packed fp6x2 value to __half2_raw.

There are C++ style classes available.

.. list-table:: C++ types
    :header-rows: 1

    * - struct
      - Notes
    * - `__hip_fp6_e2m3`
      - A single value for fp6 e2m3 type
    * - `__hip_fp6x2_e2m3`
      - A two packed value for fp4 e2m3 type
    * - `__hip_fp6x4_e2m3`
      - A four packed value for fp4 e2m3 type
    * - `__hip_fp6_e3m2`
      - A single value for fp6 e3m2 type
    * - `__hip_fp6x2_e3m2`
      - A two packed value for fp4 e3m2 type
    * - `__hip_fp6x4_e3m2`
      - A four packed value for fp4 e3m2 type

For FP6 E2M3 format.

.. code-block:: c

  struct __hip_fp6_e2m3 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP6_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp6_e2m3(const __half); // half to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const __hip_bfloat16); // bfloat16 to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const double); // double to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const float); // float to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const int); // int to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const long); // long to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const long long); // long long to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const short); // short to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const unsigned); // unsigned to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const unsigned long); // unsigned long to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const unsigned long long); // unsigned long long to fp6-e2m3
    __host__ __device__ explicit __hip_fp6_e2m3(const unsigned short); // unsigned short to fp6-e2m3

    // Operators
    // These operators are hidden if __HIP_NO_FP6_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator __half_raw(); // return a __half_raw
    __host__ __device__ operator __hip_bfloat16_raw(); // return a __hip_bfloat16_raw
    __host__ __device__ operator float(); // return a float
    __host__ __device__ operator double(); // return a double
  };

For FP6 E3M2 format.

.. code-block:: c

  struct __hip_fp6_e3m2 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP6_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp6_e3m2(const __half); // half to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const __hip_bfloat16); // bfloat16 to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const double); // double to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const float); // float to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const int); // int to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const long); // long to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const long long); // long long to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const short); // short to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const unsigned); // unsigned to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const unsigned long); // unsigned long to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const unsigned long long); // unsigned long long to fp6-e3m2
    __host__ __device__ explicit __hip_fp6_e3m2(const unsigned short); // unsigned short to fp6-e3m2

    // Operators
    // These operators are hidden if __HIP_NO_FP6_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator __half_raw(); // return a __half_raw
    __host__ __device__ operator __hip_bfloat16_raw(); // return a __hip_bfloat16_raw
    __host__ __device__ operator float(); // return a float
    __host__ __device__ operator double(); // return a double
  };


For packed fp6x2-e2m3 type

.. code-block:: c

  struct __hip_fp6x2_e2m3 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP6_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp6x2_e2m3(const __half2); // half2 to fp6x2-e2m3
    __host__ __device__ explicit __hip_fp6x2_e2m3(const __hip_bfloat162); // __hip_bfloat162 to fp6x2-e2m3
    __host__ __device__ explicit __hip_fp6x2_e2m3(const double2); // double2 to fp6x2-e2m3
    __host__ __device__ explicit __hip_fp6x2_e2m3(const float2); // float2 to fp6x2-e2m3

    // Operators
    // These operators are hidden if __HIP_NO_FP6_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator __half2_raw(); // return a __half2_raw
    __host__ __device__ operator __hip_bfloat162_raw(); // return a __hip_bfloat162_raw
    __host__ __device__ operator float2(); // return a float2
    __host__ __device__ operator double2(); // return a double2
  };

For packed fp6x2-e3m2 type

.. code-block:: c

  struct __hip_fp6x2_e3m2 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP6_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp6x2_e3m2(const __half2); // half2 to fp6x2-e3m2
    __host__ __device__ explicit __hip_fp6x2_e3m2(const __hip_bfloat162); // __hip_bfloat162 to fp6x2-e3m2
    __host__ __device__ explicit __hip_fp6x2_e3m2(const double2); // double2 to fp6x2-e3m2
    __host__ __device__ explicit __hip_fp6x2_e3m2(const float2); // float2 to fp6x2-e3m2

    // Operators
    // These operators are hidden if __HIP_NO_FP6_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator __half2_raw(); // return a __half2_raw
    __host__ __device__ operator __hip_bfloat162_raw(); // return a __hip_bfloat162_raw
    __host__ __device__ operator float2(); // return a float2
    __host__ __device__ operator double2(); // return a double2
  };

For packed FP6x4 E2M3 format.

.. code-block:: c

  struct __hip_fp6x4_e2m3 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP6_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp6x4_e2m3(const __half2, const __half2); // convert two half2 to fp6x4-e2m3
    __host__ __device__ explicit __hip_fp6x4_e2m3(const __hip_bfloat162, const __hip_bfloat162); // convert two __hip_bfloat162 to fp6x4-e2m3
    __host__ __device__ explicit __hip_fp6x4_e2m3(const double4); // double4 to fp6x4-e2m3
    __host__ __device__ explicit __hip_fp6x4_e2m3(const float4); // float4 to fp6x4-e2m3

    // Operators
    // These operators are hidden if __HIP_NO_FP6_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator float4(); // return a float4
    __host__ __device__ operator double4(); // return a double4
  };

For packed FP6x4 E3M2 format.

.. code-block:: c

  struct __hip_fp6x4_e3m2 {
    // Constructors
    // These constructors are hidden if __HIP_NO_FP6_CONVERSIONS__ is defined
    __host__ __device__ explicit __hip_fp6x4_e3m2(const __half2, const __half2); // convert two half2 to fp6x4-e3m2
    __host__ __device__ explicit __hip_fp6x4_e3m2(const __hip_bfloat162, const __hip_bfloat162); // convert two __hip_bfloat162 to fp6x4-e3m2
    __host__ __device__ explicit __hip_fp6x4_e3m2(const double4); // double4 to fp6x4-e3m2
    __host__ __device__ explicit __hip_fp6x4_e3m2(const float4); // float4 to fp6x4-e3m2

    // Operators
    // These operators are hidden if __HIP_NO_FP6_CONVERSION_OPERATORS__ is defined
    __host__ __device__ operator float4(); // return a float4
    __host__ __device__ operator double4(); // return a double4
  };


HIP Extensions
==============

HIP also provides some extensions APIs for microscaling formats. These are supported on AMD GPUs. `gfx950` provides hardware acceleration for hip extensions. Infact most APIs are 1 to 1 mapping of hardware instruction.

Scale is also an input to the APIs. Scale is defined as type `__amd_scale_t` and is of format E8M0.

hipExt Types
============

hipExt microscaling APIs introduce a bunch of types which are used throughout the set of APIs.

.. list-table:: Types
    :header-rows: 1

    * - Types
      - Notes
    * - `__amd_scale_t`
      - Store scale type which stores a value of E8M0.
    * - `__amd_fp8_storage_t`
      - Store a single fp8 value.
    * - `__amd_fp8x2_storage_t`
      - Store 2 packed fp8 value.
    * - `__amd_fp8x8_storage_t`
      - Store 8 packed fp8 value.
    * - `__amd_fp4x2_storage_t`
      - Store 2 packed fp4 value.
    * - `__amd_fp4x8_storage_t`
      - Store 8 packed fp4 value.
    * - `__amd_bf16_storage_t`
      - Store a single bf16 value.
    * - `__amd_bf16x2_storage_t`
      - Store 2 packed bf16 value.
    * - `__amd_bf16x8_storage_t`
      - Store 8 packed bf16 value.
    * - `__amd_bf16x32_storage_t`
      - Store 32 packed bf16 value.
    * - `__amd_fp16_storage_t`
      - Store a single fp16 value.
    * - `__amd_fp16x2_storage_t`
      - Store 2 packed fp16 value.
    * - `__amd_fp16x8_storage_t`
      - Store 8 packed fp16 value.
    * - `__amd_fp16x32_storage_t`
      - Store 32 packed fp16 value.
    * - `__amd_floatx2_storage_t`
      - Store 2 packed float value.
    * - `__amd_floatx8_storage_t`
      - Store 8 packed float value.
    * - `__amd_floatx16_storage_t`
      - Store 16 packed float value.
    * - `__amd_floatx32_storage_t`
      - Store 32 packed float value.
    * - `__amd_fp6x32_storage_t`
      - Store 32 packed fp6 value.
    * - `__amd_shortx2_storage_t`
      - Store 2 packed short value.

C-APIs
======

The naming style of C API is as follows:

All APIs start with `__amd`.
`_`: is used as a separator.
`cvt`: means convert i.e. convert from one format to another.
`sr`: if an API name has sr in it, means it will do stochastic rounding and will expect an input as seed.
`scale`: if an API has scale in it, means it will scale the values based on the `__amd_scale_t` input.

`create`: The following APIs will be used to create composite types from smaller values
`extract`: The following set of APIs will extract out individual values from a composite type.

Example:
`__amd_cvt_fp8x8_to_bf16x8_scale` : this API converts 8-packed fp8 values to 8 packed bf16. This will also accept input of scale to do the conversion.

`__amd_extract_fp8x2` : this API will extract out a 2 packed fp8 value from 8 packed fp8 value based on index. Example of 8-packed fp8: `{a:{fp8, fp8}, b:{fp8, fp8}, c:{fp8, fp8}, d:{fp8, fp8}}` based on index 0, 1, 2 or 3 the API will return `a`, `b`, `c` or `d` respectively.
`__amd_create_fp8x8` : this API will create 8 packed fp8 value from 4 inputs of 2 packed fp8 values.

.. list-table:: C APIs
    :header-rows: 1

    * - API
      - Notes
    * - `float __amd_cvt_fp8_to_float(const __amd_fp8_storage_t, const __amd_fp8_interpretation_t)`
      - Convert a fp8 value to float.
    * - `__amd_fp8_storage_t __amd_cvt_float_to_fp8_sr(const float, const __amd_fp8_interpretation_t, const unsigned int /* sr seed */)`
      - Convert a float to fp8 value with stochastic rounding, seed is passed as unsigned int argument.
    * - `float __amd_cvt_fp8_to_float_scale(const __amd_fp8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert a fp8 value to float with scale.
    * - `float __amd_cvt_fp8_to_float_scale(const __amd_fp8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert a fp8 value to float with scale.
    * - `__amd_floatx2_storage_t __amd_cvt_fp8x2_to_floatx2(const __amd_fp8x2_storage_t, const __amd_fp8_interpretation_t)`
      - Convert 2 packed fp8 value to 2 packed float.
    * - `__amd_fp8x2_storage_t __amd_cvt_floatx2_to_fp8x2(const __amd_floatx2_storage_t, const __amd_fp8_interpretation_t)`
      - Convert 2 packed float value to 2 packed fp8.
    * - `__amd_fp4x2_storage_t __amd_cvt_floatx2_to_fp4x2_sr_scale(const __amd_floatx2_storage_t, const __amd_fp4_interpretation_t, const unsigned int /* sr seed */, const __amd_scale_t)`
      - Convert 2 packed float value to 2 packed fp4 with stochastic rounding and scale.
    * - `__amd_floatx2_storage_t __amd_cvt_fp4x2_to_floatx2_scale(const __amd_fp4x2_storage_t , const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp4 value to 2 packed float with scale.
    * - `__amd_fp4x2_storage_t __amd_cvt_floatx2_to_fp4x2_scale(const __amd_floatx2_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed float value to 2 packed fp4 with scale.
    * - `__amd_floatx2_storage_t __amd_cvt_fp8x2_to_floatx2_scale(const __amd_fp8x2_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp8 value to 2 packed float with scale.
    * - `__amd_fp8x2_storage_t __amd_cvt_floatx2_to_fp8x2_scale(const __amd_floatx2_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed float value to 2 packed fp8 with scale.
    * - `__amd_fp6x32_storage_t __amd_cvt_bf16x32_to_fp6x32_scale(const __amd_bf16x32_storage_t, const __amd_fp6_interpretation_t, const __amd_scale_t)`
      - Convert 32 packed bf16 value to 32 packed fp6 with scale.
    * - `__amd_fp6x32_storage_t __amd_cvt_fp16x32_to_fp6x32_scale(const __amd_fp16x32_storage_t, const __amd_fp6_interpretation_t, const __amd_scale_t)`
      - Convert 32 packed fp16 value to 32 packed fp6 with scale.
    * - `__amd_fp16x2_storage_t __amd_cvt_fp8x2_to_fp16x2_scale(const __amd_fp8x2_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp8 value to 2 packed fp16 with scale.
    * - `__amd_fp16x8_storage_t __amd_cvt_fp8x8_to_fp16x8_scale(const __amd_fp8x8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp8 value to 8 packed fp16 with scale.
    * - `__amd_bf16x2_storage_t __amd_cvt_fp8x2_to_bf16x2_scale(const __amd_fp8x2_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp8 value to 2 packed bf16 with scale.
    * - `__amd_bf16x2_storage_t __amd_cvt_fp8x2_to_bf16x2_scale(const __amd_fp8x2_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp8 value to 2 packed bf16 with scale.
    * - `__amd_bf16x8_storage_t __amd_cvt_fp8x8_to_bf16x8_scale(const __amd_fp8x8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp8 value to 8 packed bf16 with scale.
    * - `__amd_fp16x32_storage_t __amd_cvt_fp6x32_to_fp16x32_scale(const __amd_fp6x32_storage_t, const __amd_fp6_interpretation_t, const __amd_scale_t)`
      - Convert 32 packed fp6 value to 32 packed fp16 with scale.
    * - `__amd_bf16x32_storage_t __amd_cvt_fp6x32_to_bf16x32_scale(const __amd_fp6x32_storage_t, const __amd_fp6_interpretation_t, const __amd_scale_t)`
      - Convert 32 packed fp6 value to 32 packed bf16 with scale.
    * - `__amd_floatx32_storage_t __amd_cvt_fp6x32_to_floatx32_scale(const __amd_fp6x32_storage_t, const __amd_fp6_interpretation_t, const __amd_scale_t)`
      - Convert 32 packed fp6 value to 32 packed float with scale.
    * - `__amd_fp16x2_storage_t __amd_cvt_fp4x2_to_fp16x2_scale(const __amd_fp4x2_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp4 value to 2 packed fp16 with scale.
    * - `__amd_fp16x8_storage_t __amd_cvt_fp4x8_to_fp16x8_scale(const __amd_fp4x8_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp4 value to 8 packed fp16 with scale.
    * - `__amd_bf16x2_storage_t __amd_cvt_fp4x2_to_bf16x2_scale(const __amd_fp4x2_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp4 value to 2 packed bf16 with scale.
    * - `__amd_bf16x8_storage_t __amd_cvt_fp4x8_to_bf16x8_scale(const __amd_fp4x8_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp4 value to 8 packed bf16 with scale.
    * - `__amd_floatx8_storage_t __amd_cvt_fp4x8_to_floatx8_scale(const __amd_fp4x8_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp4 value to 8 packed float with scale.
    * - `__amd_fp4x8_storage_t __amd_cvt_floatx8_to_fp4x8_scale(const __amd_floatx8_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed float value to 8 packed fp4 with scale.
    * - `__amd_fp8x2_storage_t __amd_cvt_fp16x2_to_fp8x2_scale(const __amd_fp16x2_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp16 value to 2 packed fp8 with scale.
    * - `__amd_fp8x2_storage_t __amd_cvt_bf16x2_to_fp8x2_scale(const __amd_bf16x2_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed bf16 value to 2 packed fp8 with scale.
    * - `__amd_fp8x8_storage_t __amd_cvt_bf16x8_to_fp8x8_scale(const __amd_bf16x8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed bf16 value to 8 packed fp8 with scale.
    * - `__amd_floatx8_storage_t __amd_cvt_fp8x8_to_floatx8_scale(const __amd_fp8x8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp8 value to 8 packed float with scale.
    * - `__amd_fp16_storage_t __amd_cvt_fp8_to_fp16_scale(const __amd_fp8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert a fp8 value to fp16 with scale.
    * - `__amd_bf16_storage_t __amd_cvt_fp8_to_bf16_scale(const __amd_fp8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert a fp8 value to bf16 with scale.
    * - `__amd_fp6x32_storage_t __amd_cvt_floatx16_floatx16_to_fp6x32_scale(const __amd_floatx16_storage_t, const __amd_floatx16_storage_t, const __amd_fp6_interpretation_t, const __amd_scale_t)`
      - Convert 2 inputs of 16-packed float values to 32 packed fp6 with scale.
    * - `__amd_fp6x32_storage_t __amd_cvt_floatx32_to_fp6x32_scale(const __amd_floatx32_storage_t, const __amd_fp6_interpretation_t, const __amd_scale_t)`
      - Convert 32 packed float values to 32 packed fp6 with scale.
    * - `__amd_fp6x32_storage_t __amd_cvt_floatx32_to_fp6x32_sr_scale(const __amd_floatx32_storage_t, const __amd_fp6_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 32 packed float values to 32 packed fp6 with stochastic rounding and scale.
    * - `__amd_fp16_storage_t __amd_cvt_float_to_fp16_sr(const float, const unsigned int)`
      - Convert a float value to fp16 with stochastic rounding.
    * - `__amd_fp16x2_storage_t __amd_cvt_float_float_to_fp16x2_sr(const float, const float, const unsigned int)`
      - Convert two inputs of float to 2 packed fp16 with stochastic rounding.
    * - `__amd_bf16_storage_t __amd_cvt_float_to_bf16_sr(const float, const unsigned int)`
      - Convert a float value to bf16 with stochastic rounding.
    * - `__amd_fp6x32_storage_t __amd_cvt_fp16x32_to_fp6x32_sr_scale(const __amd_fp16x32_storage_t, const __amd_fp6_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 32 packed fp16 values to 32 packed fp6 with stochastic rounding and scale.
    * - `__amd_fp6x32_storage_t __amd_cvt_bf16x32_to_fp6x32_sr_scale(const __amd_bf16x32_storage_t, const __amd_fp6_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 32 packed bf16 values to 32 packed fp6 with stochastic rounding and scale.
    * - `__amd_fp4x2_storage_t __amd_cvt_bf16x2_to_fp4x2_scale(const __amd_bf16x2_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed bf16 value to 2 packed fp4 with scale.
    * - `__amd_fp4x8_storage_t __amd_cvt_bf16x8_to_fp4x8_scale(const __amd_bf16x8_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed bf16 value to 8 packed fp4 with scale.
    * - `__amd_fp4x2_storage_t __amd_cvt_fp16x2_to_fp4x2_scale(const __amd_fp16x2_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 2 packed fp16 value to 2 packed fp4 with scale.
    * - `__amd_fp4x8_storage_t __amd_cvt_fp16x8_to_fp4x8_scale(const __amd_fp16x8_storage_t, const __amd_fp4_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp16 value to 8 packed fp4 with scale.
    * - `__amd_fp4x8_storage_t __amd_cvt_floatx8_to_fp4x8_sr_scale(const __amd_floatx8_storage_t, const __amd_fp4_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 8 packed float values to 8 packed fp4 with stochastic rounding and scale.
    * - `_amd_fp4x2_storage_t __amd_cvt_bf16x2_to_fp4x2_sr_scale(const __amd_bf16x2_storage_t, const __amd_fp4_interpretation_t, const unsigned int,const __amd_scale_t)`
      - Convert 2 packed bf16 value to 2 packed fp4 with stochastic rounding and scale.
    * - `__amd_fp4x8_storage_t __amd_cvt_bf16x8_to_fp4x8_sr_scale(const __amd_bf16x8_storage_t, const __amd_fp4_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 8 packed bf16 value to 8 packed fp4 with stochastic rounding and scale.
    * - `__amd_fp4x2_storage_t __amd_cvt_fp16x2_to_fp4x2_sr_scale(const __amd_fp16x2_storage_t, const __amd_fp4_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 2 packed fp16 value to 2 packed fp4 with stochastic rounding and scale.
    * - `__amd_fp4x8_storage_t __amd_cvt_fp16x8_to_fp4x8_sr_scale(const __amd_fp16x8_storage_t , const __amd_fp4_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 8 packed fp16 values to 8 packed fp4 with stochastic rounding and scale.
    * - `__amd_fp8x8_storage_t __amd_cvt_floatx8_to_fp8x8_sr_scale(const __amd_floatx8_storage_t, const __amd_fp8_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 8 packed float values to 8 packed fp8 with stochastic rounding and scale.
    * - `__amd_fp8_storage_t __amd_cvt_fp16_to_fp8_sr_scale(const __amd_fp16_storage_t, const __amd_fp8_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert a fp16 value to fp8 with stochastic rounding and scale.
    * - `__amd_fp8x8_storage_t __amd_cvt_fp16x8_to_fp8x8_sr_scale(const __amd_fp16x8_storage_t, const __amd_fp8_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 8 packed fp16 values to 8 packed fp8 with stochastic rounding and scale.
    * - `__amd_fp8_storage_t __amd_cvt_bf16_to_fp8_sr_scale(const __amd_bf16_storage_t, const __amd_fp8_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert a bf16 value to fp8 with stochastic rounding and scale.
    * - `__amd_fp8x8_storage_t __amd_cvt_bf16x8_to_fp8x8_sr_scale(const __amd_bf16x8_storage_t, const __amd_fp8_interpretation_t, const unsigned int, const __amd_scale_t)`
      - Convert 8 packed bf16 values to 8 packed fp8 with stochastic rounding and scale.
    * - `__amd_fp16_storage_t __amd_cvt_fp8_to_fp16(const __amd_fp8_storage_t, const __amd_fp8_interpretation_t)`
      - Convert a fp8 value to fp16.
    * - `__amd_fp16x2_storage_t __amd_cvt_fp8x2_to_fp16x2(const __amd_fp8x2_storage_t, const __amd_fp8_interpretation_t)`
      - Convert 2 packed fp8 value to 2 packed fp16.
    * - `__amd_fp8x2_storage_t __amd_cvt_fp16x2_to_fp8x2(const __amd_fp16x2_storage_t, const __amd_fp8_interpretation_t)`
      - Convert 2 packed fp16 value to 2 packed fp8.
    * - `__amd_fp8x8_storage_t __amd_cvt_fp16x8_to_fp8x8_scale(const __amd_fp16x8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed fp16 values to 8 packed fp8 with scale.
    * - `__amd_fp8x8_storage_t __amd_cvt_floatx8_to_fp8x8_scale(const __amd_floatx8_storage_t, const __amd_fp8_interpretation_t, const __amd_scale_t)`
      - Convert 8 packed float values to 8 packed fp8 with scale.
    * - `__amd_fp8_storage_t __amd_cvt_fp16_to_fp8_sr(const __amd_fp16_storage_t, const __amd_fp8_interpretation_t, const short)`
      - Convert a fp16 value to fp8 with stochastic rounding.
    * - `float2 __amd_cvt_floatx2_to_float2(const __amd_floatx2_storage_t)`
      - Convert 2 packed float value to hip's float2 type.
    * - `__half __amd_cvt_fp16_to_half(const __amd_fp16_storage_t)`
      - Convert fp16 type to hip's __half type.
    * - `__half2 __amd_cvt_fp16x2_to_half2(const __amd_fp16x2_storage_t)`
      - Convert 2 packed fp16 type to hip's __half2 type.
    * - `__amd_fp16_storage_t __amd_cvt_half_to_fp16(const __half)`
      - Convert hip's __half type to fp16 type.
    * - `__amd_fp16x2_storage_t __amd_cvt_half2_to_fp16x2(const __half2)`
      - Convert hip's __half2 type to 2 packed fp16.
    * - `__hip_bfloat16 __amd_cvt_bf16_to_hipbf16(const __amd_bf16_storage_t)`
      - Convert bf16 type to __hip_bfloat16 type.
    * - `__hip_bfloat162 __amd_cvt_bf16x2_to_hipbf162(const __amd_bf16x2_storage_t)`
      - Convert 2 packed bf16 type to __hip_bfloat162 type.
    * - `__amd_bf16_storage_t __amd_cvt_hipbf16_to_bf16(const __hip_bfloat16)`
      - Convert __hip_bfloat16 to bf16 type.
    * - `__amd_bf16x2_storage_t __amd_cvt_hipbf162_to_bf16x2(const __hip_bfloat162)`
      - Convert __hip_bfloat162 to 2 packed bf16 type.

HIP EXT C++ API
===============

There are C++ data structures also available. These are different from one in `<hip/hip_fp8.h>` header. These APIs expose a wider capability set which are exclusive to `gfx950`.

HIP EXT FP8 E4M3:

.. code-block:: c

  struct __hipext_ocp_fp8_e4m3  {
    // Constructor
    __host__ __device__ __hipext_ocp_fp8_e4m3(const float); // Create fp8 e4m3 from float
    __host__ __device__ __hipext_ocp_fp8_e4m3(const float, const unsigned int /* sr seed */); // Create fp8 e4m3 from float with stochastic rounding
    __host__ __device__ __hipext_ocp_fp8_e4m3(const float, const unsigned int /* sr seed */, const __amd_scale_t /* scale */); // Create fp8 e4m3 from float with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp8_e4m3(const __amd_fp16_storage_t, const unsigned int /* sr seed */, const __amd_scale_t /* scale */); // Create fp8 e4m3 from fp16 with scale
    __host__ __device__ __hipext_ocp_fp8_e4m3(const __amd_bf16_storage_t, const unsigned int /* sr seed */, const __amd_scale_t /* scale */); // Create fp8 e4m3 from bf16 with scale

    // Getters
    __host__ __device__ __amd_fp16_storage_t get_scaled_fp16(const __amd_scale_t /* scale */) const; // get scaled fp16 value
    __host__ __device__ __amd_bf16_storage_t get_scaled_bf16(const __amd_scale_t /* scale */) const; // get scaled bf16 value
    __host__ __device__ float get_scaled_float(const __amd_scale_t /* scale */) const; // get scaled float value

    // Operators
    __host__ __device__ operator float() const; // get a float value
  };

HIP EXT FP8 E5M2:

.. code-block:: c

  struct __hipext_ocp_fp8_e5m2  {
    // Constructor
    __host__ __device__ __hipext_ocp_fp8_e5m2(const float); // Create fp8 e4m3 from float
    __host__ __device__ __hipext_ocp_fp8_e5m2(const float, const unsigned int /* sr seed */); // Create fp8 e4m3 from float with stochastic rounding
    __host__ __device__ __hipext_ocp_fp8_e5m2(const float, const unsigned int /* sr seed */, const __amd_scale_t /* scale */); // Create fp8 e4m3 from float with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp8_e5m2(const __amd_fp16_storage_t, const unsigned int /* sr seed */, const __amd_scale_t /* scale */); // Create fp8 e4m3 from fp16 with scale
    __host__ __device__ __hipext_ocp_fp8_e5m2(const __amd_bf16_storage_t, const unsigned int /* sr seed */, const __amd_scale_t /* scale */); // Create fp8 e4m3 from bf16 with scale

    // Getters
    __host__ __device__ __amd_fp16_storage_t get_scaled_fp16(const __amd_scale_t /* scale */) const; // get scaled fp16 value
    __host__ __device__ __amd_bf16_storage_t get_scaled_bf16(const __amd_scale_t /* scale */) const; // get scaled bf16 value
    __host__ __device__ float get_scaled_float(const __amd_scale_t /* scale */) const; // get scaled float value

    // Operators
    __host__ __device__ operator float() const; // get a float value
  };

HIP EXT 2 Packed FP8 E4M3

.. code-block:: c

  struct __hipext_ocp_fp8x2_e4m3 {
    __host__ __device__ __hipext_ocp_fp8x2_e4m3(const float, const float); // Create fp8x2 from two floats
    __host__ __device__ __hipext_ocp_fp8x2_e4m3(const __amd_floatx2_storage_t); // Create fp8x2 from 2 packed floats
    __host__ __device__ __hipext_ocp_fp8x2_e4m3(const __amd_floatx2_storage_t, __amd_scale_t /* scale */); // Create fp8x2 from 2 packed floats with scale
    __host__ __device__ __hipext_ocp_fp8x2_e4m3(const __amd_fp16x2_storage_t, const __amd_scale_t /* scale */); // Create fp8x2 from 2 packed fp16 with scale
    __host__ __device__ __hipext_ocp_fp8x2_e4m3(const __amd_bf16x2_storage_t, const __amd_scale_t /* scale */); // Create fp8x2 from 2 packed bf16 with scale

    // Getters
    __host__ __device__ __amd_fp16x2_storage_t get_scaled_fp16x2(const __amd_scale_t) const; // Get scaled 2 packed fp16
    __host__ __device__ __amd_bf16x2_storage_t get_scaled_fp16x2(const __amd_scale_t) const; // Get scaled 2 packed fp16
    __host__ __device__ __amd_floatx2_storage_t get_scaled_floatx2(const __amd_scale_t scale)const; // Get scaled 2 packed float

    // Operators
    __host__ __device__ operator __amd_floatx2_storage_t() const; // Get 2 packed float
  };

HIP EXT 2 Packed FP8 E5M2

.. code-block:: c

  struct __hipext_ocp_fp8x2_e5m2 {
    __host__ __device__ __hipext_ocp_fp8x2_e5m2(const float, const float); // Create fp8x2 from two floats
    __host__ __device__ __hipext_ocp_fp8x2_e5m2(const __amd_floatx2_storage_t); // Create fp8x2 from 2 packed floats
    __host__ __device__ __hipext_ocp_fp8x2_e5m2(const __amd_floatx2_storage_t, __amd_scale_t /* scale */); // Create fp8x2 from 2 packed floats with scale
    __host__ __device__ __hipext_ocp_fp8x2_e5m2(const __amd_fp16x2_storage_t, const __amd_scale_t /* scale */); // Create fp8x2 from 2 packed fp16 with scale
    __host__ __device__ __hipext_ocp_fp8x2_e5m2(const __amd_bf16x2_storage_t, const __amd_scale_t /* scale */); // Create fp8x2 from 2 packed bf16 with scale

    // Getters
    __host__ __device__ __amd_fp16x2_storage_t get_scaled_fp16x2(const __amd_scale_t) const; // Get scaled 2 packed fp16
    __host__ __device__ __amd_bf16x2_storage_t get_scaled_fp16x2(const __amd_scale_t) const; // Get scaled 2 packed fp16
    __host__ __device__ __amd_floatx2_storage_t get_scaled_floatx2(const __amd_scale_t scale)const; // Get scaled 2 packed float

    // Operators
    __host__ __device__ operator __amd_floatx2_storage_t() const; // Get 2 packed float
  };

HIP EXT 32 packed FP6 E2M3

.. code-block:: c

  struct __hipext_ocp_fp6x32_e2m3 {
    __host__ __device__ __hipext_ocp_fp6x32_e2m3(const __amd_floatx16_storage_t, const __amd_floatx16_storage_t, const __amd_scale_t); // Create fp6x32 from two floatx16 with scale
    __host__ __device__ __hipext_ocp_fp6x32_e2m3(const __amd_floatx32_storage_t, const unsigned int /* seed */, const __amd_scale_t); // Create fp6x32 from two floatx32 with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp6x32_e2m3(const __amd_fp16x32_storage_t, const unsigned int /* seed */, const __amd_scale_t); // Create fp6x32 from two fp16x32 with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp6x32_e2m3(const __amd_fp16x32_storage_t, const __amd_scale_t); // Create fp6x32 from two fp16x32 with scale
    __host__ __device__ __hipext_ocp_fp6x32_e2m3(const __amd_bf16x32_storage_t, const unsigned int /* seed */, const __amd_scale_t); // Create fp6x32 from two bf16x32 with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp6x32_e2m3(const __amd_bf16x32_storage_t, const __amd_scale_t); // Create fp6x32 from two bf16x32 with scale

    // Getters
    __host__ __device__ __amd_floatx32_storage_t get_scaled_floatx32(const __amd_scale_t) const; // Get Scaled floatx32
    __host__ __device__ __amd_fp16x32_storage_t get_scaled_fp16x32(const __amd_scale_t) const; // Get Scaled fp16x32
    __host__ __device__ __amd_bf16x32_storage_t get_scaled_bf16x32(const __amd_scale_t) const; // Get Scaled bf16x32
  };

HIP EXT 32 packed FP6 E3M2

.. code-block:: c

  struct __hipext_ocp_fp6x32_e3m2 {
    __host__ __device__ __hipext_ocp_fp6x32_e3m2(const __amd_floatx16_storage_t, const __amd_floatx16_storage_t, const __amd_scale_t); // Create fp6x32 from two floatx16 with scale
    __host__ __device__ __hipext_ocp_fp6x32_e3m2(const __amd_floatx32_storage_t, const unsigned int /* seed */, const __amd_scale_t); // Create fp6x32 from two floatx32 with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp6x32_e3m2(const __amd_fp16x32_storage_t, const unsigned int /* seed */, const __amd_scale_t); // Create fp6x32 from two fp16x32 with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp6x32_e3m2(const __amd_fp16x32_storage_t, const __amd_scale_t); // Create fp6x32 from two fp16x32 with scale
    __host__ __device__ __hipext_ocp_fp6x32_e3m2(const __amd_bf16x32_storage_t, const unsigned int /* seed */, const __amd_scale_t); // Create fp6x32 from two bf16x32 with stochastic rounding and scale
    __host__ __device__ __hipext_ocp_fp6x32_e3m2(const __amd_bf16x32_storage_t, const __amd_scale_t); // Create fp6x32 from two bf16x32 with scale

    // Getters
    __host__ __device__ __amd_floatx32_storage_t get_scaled_floatx32(const __amd_scale_t) const; // Get Scaled floatx32
    __host__ __device__ __amd_fp16x32_storage_t get_scaled_fp16x32(const __amd_scale_t) const; // Get Scaled fp16x32
    __host__ __device__ __amd_bf16x32_storage_t get_scaled_bf16x32(const __amd_scale_t) const; // Get Scaled bf16x32
  };

HIP EXT 2 packed FP4

.. code-block:: c

  struct __hipext_ocp_fp4x2_e2m1 {
  __host__ __device__ __hipext_ocp_fp4x2_e2m1(const float, const float, const __amd_scale_t); // Create FP4x2 from two floats with scale
  __host__ __device__ __hipext_ocp_fp4x2_e2m1(const __amd_floatx2_storage_t, const __amd_scale_t); // Create FP4x2 from floatx2 with scale
  __host__ __device__ __hipext_ocp_fp4x2_e2m1(const __amd_bf16x2_storage_t, const __amd_scale_t); // Create FP4x2 from bf16x2 with scale
  __host__ __device__ __hipext_ocp_fp4x2_e2m1(const __amd_fp16x2_storage_t, const __amd_scale_t); // Create FP4x2 from fp16x2 with scale
  __host__ __device__ __hipext_ocp_fp4x2_e2m1(const __amd_floatx2_storage_t, const unsigned int, const __amd_scale_t); // Create FP4x2 from floatx2 with stochastic rounding and scale
  __host__ __device__ __hipext_ocp_fp4x2_e2m1(const __amd_bf16x2_storage_t, const unsigned int, const __amd_scale_t); // Create FP4x2 from bf16x2 with stochastic rounding and scale
  __host__ __device__ __hipext_ocp_fp4x2_e2m1(const __amd_fp16x2_storage_t, const unsigned int, const __amd_scale_t); // Create FP4x2 from fp16x2 with stochastic rounding and scale

  // Getters
  __host__ __device__ __amd_floatx2_storage_t get_scaled_floatx2(const __amd_scale_t) const; // get scaled floatx2
  __host__ __device__ __amd_fp16x2_storage_t get_scaled_fp16x2(const __amd_scale_t) const; // Get scaled fp16x2
  __host__ __device__ __amd_bf16x2_storage_t get_scaled_bf16x2(const __amd_scale_t) const; // Get scaled bf16x2
  };

