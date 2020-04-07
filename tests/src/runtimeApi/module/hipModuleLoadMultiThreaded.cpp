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

#include <hip/hiprtc.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <random>
#include <streambuf>
#include <experimental/filesystem>
#include <test_common.h>
#include <regex>

#define LEN 64
#define SIZE LEN << 2

#define fileName "vcpy_kernel.code"
#define kernel_name "axpy"

static constexpr auto kernel{
    R"(
#include <hip/hip_runtime.h>

__device__ float mul;

extern "C" __global__ void axpy(float* in, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = mul * in[tid] + 1.1f;
    }
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

void moduleLoadFunc(std::string filename, std::string variable) {
    float *A, *B;
    hipDeviceptr_t Ad, Bd;
    A = new float[LEN];
    B = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = 0.0f;
    }

    HIPCHECK(hipInit(0));

    // hipDevice_t device;
    // hipCtx_t context;
    // HIPCHECK(hipDeviceGet(&device, 0));
    // HIPCHECK(hipCtxCreate(&context, 0, device));

    HIPCHECK(hipMalloc((void**)&Ad, SIZE));
    HIPCHECK(hipMalloc((void**)&Bd, SIZE));
    HIPCHECK(hipMemcpyHtoD(Ad, A, SIZE));
    HIPCHECK(hipMemcpyHtoD(Bd, B, SIZE));

    hipModule_t Module;
    hipFunction_t Function;
    HIPCHECK(hipModuleLoad(&Module, filename.c_str()));

    float myDeviceGlobal_h = getRandomNumber(10, 100) * 1.1f;
    hipDeviceptr_t deviceGlobal{nullptr};
    size_t deviceGlobalSize;
    // std::cout << "Getting variable:: " << (variable.c_str() + 1) << std::endl;
    HIPCHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, variable.c_str() + 1));
    HIPCHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &myDeviceGlobal_h, deviceGlobalSize));
    HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name));

    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    struct {
        void* _Ad;
        void* _Bd;
        size_t _n;
    } args;
    args._Ad = (void*)Ad;
    args._Bd = (void*)Bd;
    args._n = LEN;
    size_t size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL, (void**)&config));

    HIPCHECK(hipStreamDestroy(stream));

    HIPCHECK(hipMemcpyDtoH(B, Bd, SIZE));

    for (uint32_t i = 0; i < LEN; i++) {
        // auto t = (myDeviceGlobal_h * A[i]) + 1.1f;
        // std::cout << t << " - " << B[i] << std::endl;
        assert((myDeviceGlobal_h * A[i] + 1.1f) == B[i]);
    }

    // HIPCHECK(hipCtxDestroy(context));
    hipFree(Ad);
    hipFree(Bd);
    delete[] A;
    delete[] B;
}

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

int main() {
    int count = getRandomNumber(4, 10);

    std::string str(kernel);

    Unique_temporary_path utp{};
    std::experimental::filesystem::create_directory(utp.path());

    std::cout << "Compiling " << count << " Kernels." << std::endl;
    std::thread* t = new std::thread[count];

    std::vector<std::pair<std::experimental::filesystem::path, std::string>> fileVarPair;

    for (int i = 0; i < count; i++) {
        std::string variable = " mul" + std::to_string(i);
        std::string tKernel = std::regex_replace(kernel, std::regex(" mul"), variable.c_str());
        // std::cout << tKernel << std::endl;
        std::string str = getCompiledCode(tKernel);
        auto tf{(utp.path() / std::to_string(i)).replace_extension(".tmp")};
        std::ofstream{tf}.write(str.data(), str.size());
        // std::cout << "FileName:: " << tf << std::endl;
        std::pair<std::experimental::filesystem::path, std::string> p;
        p.first = tf;
        p.second = variable;
        fileVarPair.emplace_back(p);
        // std::thread tmp(moduleLoadFunc, std::string(tf), varaible);
        // t[i] = std::move(tmp);
    }

    int j = 0;
    std::cout << "Launching " << count << " Threads." << std::endl;
    for (auto& i : fileVarPair) {
        std::thread tmp(moduleLoadFunc, std::string(i.first), i.second);
        t[j++] = std::move(tmp);
    }

    for (int i = 0; i < count; i++) {
        t[i].join();
        // std::cout << "Deleting:: " << (utp.path() / (std::to_string(i) + ".tmp")) << std::endl;
        std::experimental::filesystem::remove(utp.path() / (std::to_string(i) + ".tmp"));
    }
    delete[] t;

    passed();
}
