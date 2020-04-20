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

texture<float, 2, hipReadModeElementType> tex;

static constexpr auto header{
    R"(
#include <hip/hip_runtime.h>
)"};

static constexpr auto variable{
    R"(
extern "C" texture<float, 2, hipReadModeElementType> tex;
)"};

static constexpr auto kernel{
    R"(
extern "C" __global__ void __ker_name__(float* out, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    out[y * width + x] = tex2D(tex, x, y);
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

void runKernel(hipModule_t Module, int i, float* hData) {
    unsigned int width = 256;
    unsigned int height = 256;
    unsigned int size = width * height * sizeof(float);

    float* dData = NULL;
    hipMalloc((void**)&dData, size);

    struct {
        void* _Ad;
        unsigned int _Bd;
        unsigned int _Cd;
    } args;
    args._Ad = (void*) dData;
    args._Bd = width;
    args._Cd = height;

    size_t sizeTemp = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &sizeTemp, HIP_LAUNCH_PARAM_END};

    hipFunction_t Function;
    std::string kernel_name = "kernel_" + std::to_string(i);
    HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name.c_str()));

    int temp1 = width / 16;
    int temp2 = height / 16;
    HIPCHECK(
        hipModuleLaunchKernel(Function, 16, 16, 1, temp1, temp2, 1, 0, 0, NULL, (void**)&config));
    hipDeviceSynchronize();

    float* hOutputData = (float*)malloc(size);
    memset(hOutputData, 0, size);
    hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            assert(hData[i * width + j] == hOutputData[i * width + j]);
        }
    }
    hipFree(dData);
}

int main() {
    int numKernels = getRandomNumber(3, 6);
    std::string kerPlaceHolder = "kernel_", varPlaceHolder = "variable";

    std::cout << "Total Number of kernels to be generated :: " << numKernels << std::endl;

    int kernelsToRun = getRandomNumber(1, numKernels);

    std::string strFile{header};
    strFile += variable;
    for (int i = 0; i < numKernels; i++) {
        std::string tkernel{kernel}, tkname = kerPlaceHolder;
        tkname += std::to_string(i);
        tkernel = std::regex_replace(tkernel.c_str(), std::regex("__ker_name__"), tkname.c_str());
        strFile += tkernel;
    }


    std::cout << "file is:: " << std::endl << strFile << std::endl;
    std::string compiledCode = getCompiledCode(strFile);

    Unique_temporary_path utp{};
    std::experimental::filesystem::create_directory(utp.path());

    auto tf{(utp.path() / "compiledCode").replace_extension(".tmp")};
    std::ofstream{tf}.write(compiledCode.data(), compiledCode.size());

    HIPCHECK(hipInit(0));

    hipModule_t Module;
    HIPCHECK(hipModuleLoad(&Module, std::string(tf).c_str()));

    unsigned int width = 256;
    unsigned int height = 256;
    unsigned int size = width * height * sizeof(float);

    float* hData = (float*)malloc(size);
    memset(hData, 0, size);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            hData[i * width + j] = i * width + j;
        }
    }

    hipArray* array;
    HIP_ARRAY_DESCRIPTOR desc;
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = width;
    desc.Height = height;
    hipArrayCreate(&array, &desc);

    hip_Memcpy2D copyParam;
    memset(&copyParam, 0, sizeof(copyParam));
    copyParam.dstMemoryType = hipMemoryTypeArray;
    copyParam.dstArray = array;
    copyParam.srcMemoryType = hipMemoryTypeHost;
    copyParam.srcHost = hData;
    copyParam.srcPitch = width * sizeof(float);
    copyParam.WidthInBytes = copyParam.srcPitch;
    copyParam.Height = height;
    hipMemcpyParam2D(&copyParam);

    textureReference* texref;
    HIPCHECK(hipModuleGetTexRef(&texref, Module, "tex"));
    hipTexRefSetAddressMode(texref, 0, hipAddressModeWrap);
    hipTexRefSetAddressMode(texref, 1, hipAddressModeWrap);
    hipTexRefSetFilterMode(texref, hipFilterModePoint);
    hipTexRefSetFlags(texref, 0);
    hipTexRefSetFormat(texref, HIP_AD_FORMAT_FLOAT, 1);
    hipTexRefSetArray(texref, array, HIP_TRSA_OVERRIDE_FORMAT);

    std::thread* t = new std::thread[kernelsToRun];

    std::cout << "Running threads:: " << kernelsToRun << std::endl;
    for (int i = 0; i < kernelsToRun; i++) {
        int id = getRandomNumber(0, (numKernels - 1));
        std::thread tmp(runKernel, Module, id, hData);
        t[i] = std::move(tmp);
    }

    for (int i = 0; i < kernelsToRun; i++) {
        t[i].join();
    }

    std::experimental::filesystem::remove(utp.path() / (std::string("compiledCode") + ".tmp"));
    delete[] t;
    free(hData);
    hipFreeArray(array);
    passed();
}
