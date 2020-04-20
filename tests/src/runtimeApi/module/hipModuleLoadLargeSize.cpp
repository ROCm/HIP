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
 * BUILD: %t %s ../../test_common.cpp LINK_OPTIONS hiprtc stdc++fs EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include <iostream>
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <fstream>
#include <vector>
#include <thread>
#include <random>
#include <streambuf>
#include <regex>
#include <experimental/filesystem>
#include "test_common.h"


static constexpr auto header{
    R"(
#include <hip/hip_runtime.h>
)"};

static constexpr auto variable{
    R"(
__device__ int __var_name__;
)"};

static constexpr auto kernel{
    R"(
extern "C" __global__ void __ker_name__(int* in, int* out) {
    *out = __var_name__ * (*in) + __const_var__;
}
)"};

int getRandomNumber(int a, int b) {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(a, b);
    return dist6(rng);
}

class Unique_temporary_path {
    // DATA
    std::experimental::filesystem::path path_{};

   public:
    // CREATORS
    Unique_temporary_path() : path_{std::tmpnam(nullptr)} {
        while (std::experimental::filesystem::exists(path_)) {
            path_ = std::tmpnam(nullptr);
        }
    }
    Unique_temporary_path(const std::string& extension) : Unique_temporary_path{} {
        path_.replace_extension(extension);
    }

    Unique_temporary_path(const Unique_temporary_path&) = default;
    Unique_temporary_path(Unique_temporary_path&&) = default;

    ~Unique_temporary_path() noexcept { std::experimental::filesystem::remove_all(path_); }

    // MANIPULATORS
    Unique_temporary_path& operator=(const Unique_temporary_path&) = default;
    Unique_temporary_path& operator=(Unique_temporary_path&&) = default;

    // ACCESSORS
    const std::experimental::filesystem::path& path() const noexcept { return path_; }
};

std::string getCompiledCode(const std::string& program) {
    using namespace std;

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog,            // prog
                        program.c_str(),  // buffer
                        "program.cu",     // name
                        0,                // numHeaders
                        nullptr,          // headers
                        nullptr);         // includeNames

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    std::string gfxName = "gfx" + std::to_string(props.gcnArch);
    std::string sarg = "--gpu-architecture=" + gfxName;
    const char* options[] = {sarg.c_str()};

    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize = 1;
    hiprtcGetProgramLogSize(prog, &logSize);

    if (logSize) {
        string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);

        cout << log << '\n';
    }

    if (compileResult != HIPRTC_SUCCESS) {
        failed("Compilation failed.");
    }

    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);

    vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());

    hiprtcDestroyProgram(&prog);
    return string(code.begin(), code.end());
}

void runKernel(hipModule_t Module, int i, int a, int b) {
    int *A, *B;
    hipDeviceptr_t Ad, Bd;
    A = new int;
    B = new int;

    *A = getRandomNumber(10, 100);
    *B = 0;

    HIPCHECK(hipMalloc((void**)&Ad, sizeof(int)));
    HIPCHECK(hipMalloc((void**)&Bd, sizeof(int)));
    HIPCHECK(hipMemcpyHtoD(Ad, A, sizeof(int)));
    HIPCHECK(hipMemcpyHtoD(Bd, B, sizeof(int)));

    hipFunction_t Function;
    std::string kernel_name = "kernel_" + std::to_string(i);
    HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name.c_str()));

    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    struct {
        void* _Ad;
        void* _Bd;
    } args;
    args._Ad = (void*)Ad;
    args._Bd = (void*)Bd;
    size_t size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, 1, 1, 1, 0, stream, NULL, (void**)&config));

    HIPCHECK(hipStreamDestroy(stream));

    HIPCHECK(hipMemcpyDtoH(B, Bd, sizeof(int)));

    assert(*B == (*A * a + b));
    delete A;
    delete B;
    hipFree(Ad);
    hipFree(Bd);
}

int main() {
    int numKernels = getRandomNumber(30, 60);
    std::string kerPlaceHolder = "kernel_", varPlaceHolder = "variable";

    std::vector<int> constVars;
    constVars.reserve(numKernels);

    std::cout << "Total Number of kernels to be generated :: " << numKernels << std::endl;

    int kernelsToRun = getRandomNumber(10, numKernels);

    std::string strFile{header};
    strFile += variable;
    for (int i = 0; i < numKernels; i++) {
        std::string tkernel{kernel}, tkname = kerPlaceHolder;
        tkname += std::to_string(i);
        tkernel = std::regex_replace(tkernel.c_str(), std::regex("__ker_name__"), tkname.c_str());
        int constNum = getRandomNumber(10, 1000);
        constVars.push_back(constNum);
        tkernel = std::regex_replace(tkernel.c_str(), std::regex("__const_var__"),
                                     std::to_string(constNum).c_str());
        strFile += tkernel;
    }
    strFile =
        std::regex_replace(strFile.c_str(), std::regex("__var_name__"), varPlaceHolder.c_str());

    // std::cout << "file is:: " << std::endl << strFile << std::endl;
    std::string compiledCode = getCompiledCode(strFile);

    Unique_temporary_path utp{};
    std::experimental::filesystem::create_directory(utp.path());

    auto tf{(utp.path() / "compiledCode").replace_extension(".tmp")};
    std::ofstream{tf}.write(compiledCode.data(), compiledCode.size());

    HIPCHECK(hipInit(0));

    hipModule_t Module;
    HIPCHECK(hipModuleLoad(&Module, std::string(tf).c_str()));

    int myDeviceGlobal_h = getRandomNumber(10, 10000);
    hipDeviceptr_t deviceGlobal;
    size_t deviceGlobalSize;
    HIPCHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, varPlaceHolder.c_str()));
    HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &myDeviceGlobal_h, deviceGlobalSize));

    std::thread* t = new std::thread[kernelsToRun];

    std::cout << "Running threads:: " << kernelsToRun << std::endl;
    for (int i = 0; i < kernelsToRun; i++) {
        int id = getRandomNumber(0, (numKernels - 1));
        std::thread tmp(runKernel, Module, id, myDeviceGlobal_h, constVars[id]);
        t[i] = std::move(tmp);
    }

    for (int i = 0; i < kernelsToRun; i++) {
        t[i].join();
    }

    std::experimental::filesystem::remove(utp.path() / (std::string("compiledCode") + ".tmp"));
    delete[] t;
    passed();
}
