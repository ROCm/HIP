/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/* HIT_START
 * BUILD: %t %s ../test_common.cpp LINK_OPTIONS hiprtc EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include <test_common.h>

#define HIPRTC_GET_TYPE_NAME
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <string>
#include <vector>

static constexpr auto gpu_program{
R"(
#include <hip/hip_runtime.h>

namespace N1 { struct S1_t { int i; double d; }; }
template<typename T>
__global__ void f3(int *result) { *result = sizeof(T); }
)"};

// note: this structure is also defined in GPU code string. Should ideally
// be in a header file included by both GPU code string and by CPU code.
namespace N1 { struct S1_t { int i; double d; }; };

template <typename T>
std::string getKernelNameForType(void)
{
    std::string type_name;
    hiprtcGetTypeName<T>(&type_name);
    return std::string{"f3<"} + type_name + '>';
}

int main()
{
    using namespace std;

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, gpu_program, "gpu_program.cu", 0, nullptr,
                        nullptr);

    vector<string> name_vec;
    vector<int> expected_result;

    name_vec.push_back(getKernelNameForType<int>());
    expected_result.push_back(sizeof(int));
    name_vec.push_back(getKernelNameForType<double>());
    expected_result.push_back(sizeof(double));
    name_vec.push_back(getKernelNameForType<N1::S1_t>());
    expected_result.push_back(sizeof(N1::S1_t));

    for (auto&& x : name_vec) hiprtcAddNameExpression(prog, x.c_str());

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    std::string gfxName = "gfx" + std::to_string(props.gcnArch);
    std::string sarg = "--gpu-architecture=" + gfxName;
    const char* options[] = {
            sarg.c_str()
    };

    hiprtcResult compileResult = hiprtcCompileProgram(prog, 1, options);

    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);

    if (logSize) {
        string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);

        cout << log << '\n';
    }

    if (compileResult != HIPRTC_SUCCESS) { failed("Compilation failed."); }

    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);

    vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());

    hipModule_t module;
    hipModuleLoadDataEx(&module, code.data(), 0, nullptr, nullptr);

    hipDeviceptr_t dResult;
    int hResult = 0;
    hipMalloc(&dResult, sizeof(hResult));
    hipMemcpyHtoD(dResult, &hResult, sizeof(hResult));

    for (size_t i = 0; i < name_vec.size(); ++i) {
        const char *name;
        hiprtcGetLoweredName(prog, name_vec[i].c_str(), &name);

        hipFunction_t kernel;
        hipModuleGetFunction(&kernel, module, name);

        struct { hipDeviceptr_t a_; } args{dResult};

        auto size = sizeof(args);
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                          HIP_LAUNCH_PARAM_END};

        hipModuleLaunchKernel(kernel,
                              1, 1, 1,
                              1, 1, 1,
                              0, nullptr,
                              nullptr, config);

        hipMemcpyDtoH(&hResult, dResult, sizeof(hResult));

        if (expected_result[i] != hResult) { failed("Validation failed."); }
    }

    hipFree(dResult);
    hipModuleUnload(module);

    hiprtcDestroyProgram(&prog);

    passed();
}
