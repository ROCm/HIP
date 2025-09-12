// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// [sphinx-start]
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define hip_check(hip_call)                                                      \
{                                                                                \
    auto hip_res = hip_call;                                                     \
    if (hip_res != hipSuccess)                                                   \
    {                                                                            \
        std::cerr << "Failed in HIP call: " << #hip_call                         \
                  << " at " << __FILE__ << ":" << __LINE__                       \
                  << " with error: " << hipGetErrorString(hip_res) << std::endl; \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
}

__device__ __hip_fp8_storage_t d_convert_float_to_fp8(float in, __hip_fp8_interpretation_t interpret, __hip_saturation_t sat)
{
    return __hip_cvt_float_to_fp8(in, sat, interpret);
}

__global__ void float_to_fp8_to_float(float *in, __hip_fp8_interpretation_t interpret, __hip_saturation_t sat, float *out, size_t size)
{
    int i = threadIdx.x;
    if (i < size)
    {
        auto fp8 = d_convert_float_to_fp8(in[i], interpret, sat);
        // Implicit conversion from fp8 to float is defined by HIP runtime
        out[i] = fp8;
    }
}

__hip_fp8_storage_t convert_float_to_fp8(float in,                             /* Input val */
                                         __hip_fp8_interpretation_t interpret, /* interpretation of number E4M3/E5M2 */
                                         __hip_saturation_t sat                /* Saturation behavior */
                                        )
{
    return __hip_cvt_float_to_fp8(in, sat, interpret);
}

int main()
{
    constexpr size_t size = 32;
    hipDeviceProp_t prop;
    hip_check(hipGetDeviceProperties(&prop, 0));
    bool is_supported = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos); // gfx94x
    if(!is_supported)
    {
        std::cerr << "Need a gfx94x, but found: " << prop.gcnArchName << std::endl;
        std::cerr << "No device conversions are supported, only host conversions are supported." << std::endl;
        return EXIT_SUCCESS;
    }

    const __hip_fp8_interpretation_t interpret = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos)
                                                    ? __HIP_E4M3_FNUZ // gfx94x
                                                    : __HIP_E4M3;
    constexpr __hip_saturation_t sat = __HIP_SATFINITE;

    std::vector<float> in;
    in.reserve(size);
    for (size_t i = 0; i < size; i++)
        in.push_back(i + 1.1f);

    std::cout << "Converting float to fp8 and back..." << std::endl;
    // CPU convert
    std::vector<float> cpu_out;
    cpu_out.reserve(size);
    for (const auto &fval : in)
    {
        auto fp8 = convert_float_to_fp8(fval, interpret, sat);
        // Implicit conversion from fp8 to float is defined by HIP runtime
        cpu_out.push_back(fp8);
    }

    // GPU convert
    float *d_in, *d_out;
    hip_check(hipMalloc(&d_in, sizeof(float) * size));
    hip_check(hipMalloc(&d_out, sizeof(float) * size));

    hip_check(hipMemcpy(d_in, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));

    float_to_fp8_to_float<<<1, size>>>(d_in, interpret, sat, d_out, size);

    std::vector<float> gpu_out(size, 0.0f);
    hip_check(hipMemcpy(gpu_out.data(), d_out, sizeof(float) * gpu_out.size(), hipMemcpyDeviceToHost));

    hip_check(hipFree(d_in));
    hip_check(hipFree(d_out));

    // Validation
    for (size_t i = 0; i < size; i++)
    {
        if (cpu_out[i] != gpu_out[i])
        {
            std::cerr << "cpu round trip result: " << cpu_out[i]
                      << " - gpu round trip result: " << gpu_out[i] << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "...CPU and GPU round trip convert matches." << std::endl;

    return EXIT_SUCCESS;
}
// [sphinx-end]
