.. meta::
    :description: This page describes FP8 numbers present in HIP.
    :keywords: AMD, ROCm, HIP, fp8, fnuz, ocp

*******************************************************************************
FP8 Numbers
*******************************************************************************

`FP8 numbers <https://arxiv.org/pdf/2209.05433>`_ were introduced to accelerate deep learning inferencing. They provide higher throughput of matrix operations because the smaller size allows more of them in the available fixed memory.

HIP has two FP8 number representations called *FP8-OCP* and *FP8-FNUZ*.

Open Compute Project(OCP) number definition can be found `here <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>`_.

Definition of FNUZ: fnuz suffix means only finite and NaN values are supported. Unlike other types, Inf are not supported.
NaN is when sign bit is set and all other exponent and mantissa bits are 0. All other values are finite.
This provides one extra value of exponent and adds to the range of supported FP8 numbers.

FP8 Definition
==============

FP8 numbers are composed of a sign, an exponent and a mantissa. Their sizes are dependent on the format.
There are two formats of FP8 numbers, E4M3 and E5M2.

- E4M3: 1 bit sign, 4 bit exponent, 3 bit mantissa
- E5M2: 1 bit sign, 5 bit exponent, 2 bit mantissa

HIP Header
==========

The `HIP header <https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_hip_fp8.h>`_ defines the FP8 ocp/fnuz numbers.

Supported Devices
=================

.. list-table:: Supported devices for fp8 numbers
    :header-rows: 1

    * - Device Type
      - FNUZ FP8
      - OCP FP8
    * - Host
      - Yes
      - Yes
    * - gfx940/gfx941/gfx942
      - Yes
      - No

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
