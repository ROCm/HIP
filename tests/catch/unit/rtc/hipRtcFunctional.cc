/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

/* This test verifies few functional scenarios of hiprtc
 * hipRTC program should be created even if the name passed is empty or null
 * hipRTC program compilation should succeed even if gpu arch is not specified in the options
 * hipRTC should be able to compile kernels using  __forceinline__ keyword
 */
#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

const char* funname = "testinline";
static constexpr auto code {
R"(
__forceinline__ __device__ float f() { return 123.4f; }
extern "C"
__global__ void testinline()
{
 f();
}
)"};

TEST_CASE("Unit_hiprtc_functional") {
  using namespace std;
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 0, 0)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

#if HT_NVIDIA
  int device = 0;
  HIPCHECK(hipInit(0));
  hipCtx_t ctx;
  HIPCHECK(hipCtxCreate(&ctx, 0, device));
#endif

  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, funname));

  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, 0, nullptr, 0));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipModuleUnload(module));

#if HT_NVIDIA
  HIPCHECK(hipCtxDestroy(ctx));
#endif
}
