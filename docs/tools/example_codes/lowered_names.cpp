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

// [sphinx-source-start]
static constexpr const char gpu_program[] {
R"(
    __device__ int V1; // set from host code
    static __global__ void f1(int *result)
    {
        *result = V1 + 10;
    }

    namespace N1
    {
        namespace N2
        {
            __constant__ int V2; // set from host code
            __global__ void f2(int *result)
            {
                *result = V2 + 20;
            }
        }
    }

    template<typename T>
    __global__ void f3(int *result)
    {
        *result = sizeof(T);
    }
)"};
// [sphinx-source-end]

int main()
{
    using namespace std::string_literals;

    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, gpu_program, "gpu_source.cpp", 0, nullptr, nullptr));

    std::vector<std::string> kernel_names;
    std::vector<std::string> variable_names;
    std::vector<int> initial_values;
    std::vector<int> expected_results;
    initial_values.emplace_back(100);
    initial_values.emplace_back(200);
    expected_results.emplace_back(110);
    expected_results.emplace_back(220);
    expected_results.emplace_back(static_cast<int>(sizeof(int)));

    // [sphinx-add-expression-start]
    kernel_names.emplace_back("&f1"s);
    kernel_names.emplace_back("N1::N2::f2"s);
    kernel_names.emplace_back("f3<int>"s);
    for(auto&& name : kernel_names)
        HIPRTC_CHECK(hiprtcAddNameExpression(prog, name.c_str()));

    variable_names.emplace_back("&V1"s);
    variable_names.emplace_back("&N1::N2::V2");
    for(auto&& name : variable_names)
        HIPRTC_CHECK(hiprtcAddNameExpression(prog, name.c_str()));
    // [sphinx-add-expression-end]

    hipDeviceProp_t props;
    int device = 0;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    auto sarg = std::string{"--gpu-architecture="} + props.gcnArchName;  // device for which binary is to be generated

    const char* options[] = {sarg.c_str()};

    HIPRTC_CHECK(hiprtcCompileProgram(prog, 1, options));

    std::size_t logSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
    if (logSize)
    {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        std::cerr << "Compilation failed or produced warnings: " << log << std::endl;
        std::abort();
    }

    std::size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

    std::vector<char> kernel_binary(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, kernel_binary.data()));

    std::vector<std::string> lowered_kernel_names;
    std::vector<std::string> lowered_variable_names;
    // [sphinx-get-kernel-name-start]
    for(auto&& name : kernel_names)
    {
        const char* lowered_name = nullptr;
        HIPRTC_CHECK(hiprtcGetLoweredName(prog, name.c_str(), &lowered_name));
        lowered_kernel_names.emplace_back(lowered_name);
    }
    // [sphinx-get-kernel-name-end]
    // [sphinx-get-variable-name-start]
    for(auto&& name : variable_names)
    {
        const char* lowered_name = nullptr;
        HIPRTC_CHECK(hiprtcGetLoweredName(prog, name.c_str(), &lowered_name));
        lowered_variable_names.emplace_back(lowered_name);
    }
    // [sphinx-get-variable-name-end]

    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    hipModule_t module;

    HIP_CHECK(hipModuleLoadData(&module, kernel_binary.data()));

    for(auto i = std::size_t{0}; i < initial_values.size(); ++i)
    {
        auto name = lowered_variable_names.at(i);
        auto initial_value = initial_values.at(i);

        // [sphinx-update-variable-start]
        hipDeviceptr_t variable_addr;
        std::size_t bytes{};
        HIP_CHECK(hipModuleGetGlobal(&variable_addr, &bytes, module, name.c_str()));
        HIP_CHECK(hipMemcpyHtoD(variable_addr, &initial_value, sizeof(initial_value)));
        // [sphinx-update-variable-end]
    }

    hipDeviceptr_t d_result;
    auto h_result = 0;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(h_result)));
    HIP_CHECK(hipMemcpyHtoD(d_result, &h_result, sizeof(h_result)));

    struct
    {
        hipDeviceptr_t ptr;
    } args{d_result};
    auto args_size = sizeof(args);
    
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                      HIP_LAUNCH_PARAM_END};

    for(auto i = std::size_t{0}; i < lowered_kernel_names.size(); ++i)
    {
        auto name = lowered_kernel_names.at(i);
        auto expected = expected_results.at(i);
        // [sphinx-launch-kernel-start]
        hipFunction_t kernel;
        HIP_CHECK(hipModuleGetFunction(&kernel, module, name.c_str()));
        HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config));
        // [sphinx-launch-kernel-end]
        HIP_CHECK(hipMemcpyDtoH(&h_result, d_result, sizeof(h_result)));
        if(expected != h_result)
        {
            std::cerr << "Validation failed. expected = " << expected << ", h_result = " << h_result << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Validation passed." << std::endl;

    HIP_CHECK(hipFree(reinterpret_cast<void*>(d_result)));

    return EXIT_SUCCESS;
}
