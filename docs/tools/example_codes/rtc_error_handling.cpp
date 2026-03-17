// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
