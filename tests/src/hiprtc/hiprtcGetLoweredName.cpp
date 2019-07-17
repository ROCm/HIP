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
 * BUILD: %t %s ../test_common.cpp LINK_OPTIONS hiprtc EXCLUDE_HIP_PLATFORM all
 * TEST: %t
 * HIT_END
 */
#include <test_common.h>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <string>
#include <vector>


static constexpr const char gpu_program[]{
R"(
#include <hip/hip_runtime.h>

__device__ int V1; // set from host code
static __global__ void f1(int *result) { *result = V1 + 10; }
namespace N1 {
namespace N2 {
__constant__ int V2; // set from host code
__global__ void f2(int *result) { *result = V2 + 20; }
}
}
template<typename T>
__global__ void f3(int *result) { *result = sizeof(T); }
)"};

int main()
{
    using namespace std;

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, gpu_program, "prog.cu", 0, nullptr, nullptr);

    vector<string> kernel_name_vec;
    vector<string> variable_name_vec;
    vector<int> variable_initial_value;
    vector<int> expected_result;

    kernel_name_vec.push_back("&f1");
    expected_result.push_back(10 + 100);
    kernel_name_vec.push_back("N1::N2::f2");
    expected_result.push_back(20 + 200);
    kernel_name_vec.push_back("f3<int>");
    expected_result.push_back(sizeof(int));
    kernel_name_vec.push_back("f3<double>");
    expected_result.push_back(sizeof(double));

    for (auto&& x : kernel_name_vec) hiprtcAddNameExpression(prog, x.c_str());

    variable_name_vec.push_back("&V1");
    variable_initial_value.push_back(100);
    variable_name_vec.push_back("&N1::N2::V2");
    variable_initial_value.push_back(200);

    for (auto&& x : variable_name_vec) hiprtcAddNameExpression(prog, x.c_str());

    hiprtcResult compileResult = hiprtcCompileProgram(prog, 0, nullptr);

    // Obtain compilation log from the program.
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
    hipModuleLoadData(&module, code.data());

    hipDeviceptr_t dResult;
    int hResult = 0;
    hipMalloc(&dResult, sizeof(hResult));
    hipMemcpyHtoD(dResult, &hResult, sizeof(hResult));

    for (decltype(variable_name_vec.size()) i = 0; i != variable_name_vec.size(); ++i) {
        const char* name;
        hiprtcGetLoweredName(prog, variable_name_vec[i].c_str(), &name);

        int initial_value = variable_initial_value[i];

        hipDeviceptr_t variable_addr;
        size_t bytes{};
        hipModuleGetGlobal(&variable_addr, &bytes, module, name);
        hipMemcpyHtoD(variable_addr, &initial_value, sizeof(initial_value));
    }

    for (decltype(kernel_name_vec.size()) i = 0; i != kernel_name_vec.size(); ++i) {
        const char* name;
        hiprtcGetLoweredName(prog, kernel_name_vec[i].c_str(), &name);

        hipFunction_t kernel;
        hipModuleGetFunction(&kernel, module, name);

        struct { hipDeviceptr_t a_; } args{dResult};

        auto size = sizeof(args);
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                          HIP_LAUNCH_PARAM_END};

        hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                              config);

        hipMemcpyDtoH(&hResult, dResult, sizeof(hResult));

        if (expected_result[i] != hResult) { failed("Validation failed."); }
    }

    hipFree(dResult);
    hipModuleUnload(module);

    hiprtcDestroyProgram(&prog);

    passed();
}
