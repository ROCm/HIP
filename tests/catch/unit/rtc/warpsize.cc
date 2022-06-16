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

#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

const char* funcname = "getWarpSize";
static constexpr auto code{
    R"(
extern "C"
__global__
void getWarpSize(int* warpSizePtr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) *warpSizePtr = warpSize;
}
)"};

TEST_CASE("Unit_hiprtc_warpsize") {
  using namespace std;
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, code, "code.cu", 0, nullptr, nullptr));

  hipDeviceProp_t props;
  int device = 0;
  hipGetDeviceProperties(&props, device);
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
#else
  std::string sarg = std::string("--gpu-architecture=compute_")
    + std::to_string(props.major) + std::to_string(props.minor);
#endif
  vector<const char*> opts;
  opts.push_back(sarg.c_str());

  hiprtcResult compileResult{hiprtcCompileProgram(prog, opts.size(), opts.data())};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  int* d_warpSize;
  HIP_CHECK(hipMalloc(&d_warpSize, sizeof(int)));

  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, funcname));

  void* args[] = { &d_warpSize };
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, 0, args, 0));
  HIP_CHECK(hipDeviceSynchronize());

  int h_warpSize;
  HIP_CHECK(hipMemcpyDtoH(&h_warpSize, reinterpret_cast<hipDeviceptr_t>(d_warpSize), sizeof(int)));
  HIP_CHECK(hipFree(d_warpSize));
  HIP_CHECK(hipModuleUnload(module));
  // Verifies warp size returned by the kernel via hiprtc and runtime to be same
  REQUIRE(h_warpSize == props.warpSize);

}
