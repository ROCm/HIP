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

// source code for hiprtc
static constexpr auto kernel_source{
    R"(
    extern "C"
    __global__ void vector_add(float* output, float* input1, float* input2, size_t size)
    {
        int i = threadIdx.x;
        if (i < size)
        {
            output[i] = input1[i] + input2[i];
        }
    }
)"};

int main()
{
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

    // [sphinx-options-start]
    auto sarg = std::string{"-fgpu-rdc"};
    const char* compile_options[] = {sarg.c_str()};

    rtc_ret_code = hiprtcCompileProgram(prog,      // hiprtcProgram
                                        1,         // Number of options
                                        compile_options);
    // [sphinx-options-end]
    if (rtc_ret_code != HIPRTC_SUCCESS)
    {
        std::cerr << "Failed to create program" << std::endl;
        std::abort();
    }

    std::size_t logSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));

    if (logSize)
    {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        std::cerr << "Compilation failed or produced warnings: " << log << std::endl;
        std::abort();
    }

    // [sphinx-bitcode-start]
    std::size_t bitCodeSize;
    HIPRTC_CHECK(hiprtcGetBitcodeSize(prog, &bitCodeSize));

    std::vector<char> kernel_bitcode(bitCodeSize);
    HIPRTC_CHECK(hiprtcGetBitcode(prog, kernel_bitcode.data()));
    // [sphinx-bitcode-end]

    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    // [sphinx-link-create-start]
    const char* isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm", "-inlinehint-threshold=1"};
    std::vector<hiprtcJIT_option> jit_options = {HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
                                                 HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
    std::size_t isaoptssize = 4;
    void* lopts[] = {reinterpret_cast<void*>(isaopts),
                     reinterpret_cast<void*>(isaoptssize)};
    hiprtcLinkState linkstate;
    HIPRTC_CHECK(hiprtcLinkCreate(2u, jit_options.data(), reinterpret_cast<void**>(lopts), &linkstate));
    // [sphinx-link-create-end]

    auto input_type = HIPRTC_JIT_INPUT_LLVM_BITCODE;
    auto bit_code_ptr = kernel_bitcode.data();
    auto bit_code_size = bitCodeSize;
    // [sphinx-link-add-start]
    HIPRTC_CHECK(hiprtcLinkAddData(linkstate,        // HIPRTC link state
                                   input_type,            // type of the input data or bitcode
                                   bit_code_ptr,          // input data which is null terminated
                                   bit_code_size,         // size of the input data
                                   "a",                   // optional name for this input
                                   0,                     // size of the options
                                   nullptr,               // Array of options applied to this input
                                   nullptr));             // Array of option values cast to void*
    // [sphinx-link-add-end]

    void* binary = nullptr;
    auto binarySize = std::size_t{};
    // [sphinx-link-complete-start]
    HIPRTC_CHECK(hiprtcLinkComplete(linkstate,       // HIPRTC link state
                                  &binary,         // upon success, points to the output binary
                                  &binarySize));   // size of the binary is stored (optional)
    // [sphinx-link-complete-end]

    hipModule_t module;
    hipFunction_t kernel;

    HIP_CHECK(hipModuleLoadData(&module, binary));
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "vector_add"));

    HIPRTC_CHECK(hiprtcLinkDestroy(linkstate));

    constexpr std::size_t ele_size = 256;  // total number of items to add
    std::vector<float> hinput, output;
    hinput.reserve(ele_size);
    output.reserve(ele_size);
    for (std::size_t i = 0; i < ele_size; i++)
    {
        hinput.push_back(static_cast<float>(i + 1));
        output.push_back(0.0f);
    }

    float *dinput1, *dinput2, *doutput;
    HIP_CHECK(hipMalloc(&dinput1, sizeof(float) * ele_size));
    HIP_CHECK(hipMalloc(&dinput2, sizeof(float) * ele_size));
    HIP_CHECK(hipMalloc(&doutput, sizeof(float) * ele_size));

    HIP_CHECK(hipMemcpy(dinput1, hinput.data(), sizeof(float) * ele_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dinput2, hinput.data(), sizeof(float) * ele_size, hipMemcpyHostToDevice));

    struct
    {
        float* output;
        float* input1;
        float* input2;
        std::size_t size;
    } args{doutput, dinput1, dinput2, ele_size};

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, ele_size, 1, 1, 0, nullptr, nullptr, config));

    HIP_CHECK(hipMemcpy(output.data(), doutput, sizeof(float) * ele_size, hipMemcpyDeviceToHost));

    for (std::size_t i = 0; i < ele_size; i++)
    {
        if ((hinput[i] + hinput[i]) != output[i])
        {
            std::cout << "Failed in validation: " << (hinput[i] + hinput[i]) << " - " << output[i] << std::endl;
            std::abort();
        }
    }
    std::cout << "Passed" << std::endl;

    HIP_CHECK(hipFree(dinput1));
    HIP_CHECK(hipFree(dinput2));
    HIP_CHECK(hipFree(doutput));

    return EXIT_SUCCESS;
}
// [sphinx-stop]
