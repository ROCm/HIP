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

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>

static constexpr auto NUM_THREADS{128};
static constexpr auto NUM_BLOCKS{32};

static constexpr auto saxpy{
R"(
#include <hip/hip_runtime.h>

extern "C"
__global__
void saxpy(float a, float* x, float* y, float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
       out[tid] = a * x[tid] + y[tid];
    }
}
)"};

int main()
{
    using namespace std;

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog,      // prog
                        saxpy,      // buffer
                        "saxpy.cu", // name
                        0,          // numHeaders
                        nullptr,    // headers
                        nullptr);   // includeNames

    hiprtcResult compileResult{hiprtcCompileProgram(prog, 0, nullptr)};

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

    hiprtcDestroyProgram(&prog);

    hipModule_t module;
    hipFunction_t kernel;

    hipModuleLoadData(&module, code.data());
    hipModuleGetFunction(&kernel, module, "saxpy");

    size_t n = NUM_THREADS * NUM_BLOCKS;
    size_t bufferSize = n * sizeof(float);

    float a = 5.1f;
    unique_ptr<float[]> hX{new float[n]};
    unique_ptr<float[]> hY{new float[n]};
    unique_ptr<float[]> hOut{new float[n]};

    for (size_t i = 0; i < n; ++i) {
        hX[i] = static_cast<float>(i);
        hY[i] = static_cast<float>(i * 2);
    }

    hipDeviceptr_t dX, dY, dOut;
    hipMalloc(&dX, bufferSize);
    hipMalloc(&dY, bufferSize);
    hipMalloc(&dOut, bufferSize);
    hipMemcpyHtoD(dX, hX.get(), bufferSize);
    hipMemcpyHtoD(dY, hY.get(), bufferSize);

    struct {
        float a_;
        hipDeviceptr_t b_;
        hipDeviceptr_t c_;
        hipDeviceptr_t d_;
        size_t e_;
    } args{a, dX, dY, dOut, n};

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    hipModuleLaunchKernel(kernel, NUM_BLOCKS, 1, 1, NUM_THREADS, 1, 1,
                          0, nullptr, nullptr, config);

    hipMemcpyDtoH(hOut.get(), dOut, bufferSize);

    for (size_t i = 0; i < n; ++i) {
       if (a * hX[i] + hY[i] != hOut[i]) { failed("Validation failed."); }
    }

    hipFree(dX);
    hipFree(dY);
    hipFree(dOut);

    hipModuleUnload(module);

    passed();
}
