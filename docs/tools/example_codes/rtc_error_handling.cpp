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

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_RET_CODE(call, ret_code)                                                             \
{                                                                                                  \
    if ((call) != ret_code)                                                                        \
    {                                                                                              \
        std::cout << "Failed in call: " << #call << std::endl;                                     \
        std::abort();                                                                              \
    }                                                                                              \
}
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)
#define HIPRTC_CHECK(call) CHECK_RET_CODE(call, HIPRTC_SUCCESS)

int main()
{
    const char* kernel_source = "adafsfgadascvsfgsadfbdt";
    hiprtcProgram prog;
    auto rtc_ret_code = hiprtcCreateProgram(&prog,            // HIPRTC program handle
                                            kernel_source,    // kernel source string
                                            "vector_add.cpp", // Name of the file
                                            0,                // Number of headers
                                            nullptr,          // Header sources
                                            nullptr);         // Name of header file

    if (rtc_ret_code != HIPRTC_SUCCESS)
    {
        std::cerr << "Failed to create program" << std::endl;
        std::abort();
    }

    hipDeviceProp_t props;
    int device = 0;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    auto sarg = std::string{"--gpu-architecture="} + props.gcnArchName;  // device for which binary is to be generated

    const char* opts[] = {sarg.c_str()};

    // [sphinx-start]
    hiprtcResult result;
    result = hiprtcCompileProgram(prog, 1, opts);
    if (result != HIPRTC_SUCCESS)
    {
        std::cout << "hiprtcCompileProgram fails with error " << hiprtcGetErrorString(result);
    }
    // [sphinx-end]

    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    return EXIT_SUCCESS;
}
