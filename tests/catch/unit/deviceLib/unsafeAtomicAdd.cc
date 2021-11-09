/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <cmath>

static constexpr auto kernel{
    R"(
extern "C"
__global__
void unsafeAdd_f(float *p, float v)
{
    auto val = unsafeAtomicAdd(p, v);
}

extern "C"
__global__
void unsafeAdd_d(double *p, double v)
{
    auto val = unsafeAtomicAdd(p, v);
}
)"};


TEST_CASE("Unit_unsafeAtomicAdd") {
  using namespace std;
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if (gfxName == "gfx90a" || gfxName.find("gfx90a:") == 0) {
    hiprtcProgram prog;
    hiprtcCreateProgram(&prog,        // prog
                        kernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
    const char* options[] = {sarg.c_str()};
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

    size_t logSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
    if (logSize) {
      string log(logSize, '\0');
      HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
      INFO(log);
    }

    REQUIRE(compileResult == HIPRTC_SUCCESS);
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

    vector<char> code(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    float* fX;
    double* dX;
    HIP_CHECK(hipMalloc(&fX, sizeof(float)));
    HIP_CHECK(hipMalloc(&dX, sizeof(double)));

    hipModule_t module;
    hipFunction_t f_kernel, d_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "unsafeAdd_f"));
    HIP_CHECK(hipModuleGetFunction(&d_kernel, module, "unsafeAdd_d"));

    float f_val = 10.1f;
    double d_val = 10.1;

    HIP_CHECK(hipMemcpy(fX, &f_val, sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, &d_val, sizeof(double), hipMemcpyHostToDevice));

    struct {
      float* p;
      float val;
    } args_f{fX, f_val};

    struct {
      double* p;
      double val;
    } args_d{dX, d_val};

    auto size_f = sizeof(args_f);
    void* config_f[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                        &size_f, HIP_LAUNCH_PARAM_END};

    auto size_d = sizeof(args_d);
    void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_d, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                        &size_d, HIP_LAUNCH_PARAM_END};

    hipModuleLaunchKernel(f_kernel, 10, 1, 1, 100, 1, 1, 0, nullptr, nullptr, config_f);
    hipModuleLaunchKernel(d_kernel, 10, 1, 1, 100, 1, 1, 0, nullptr, nullptr, config_d);

    float res_f = 0.0f;
    double res_d = 0.0;
    HIP_CHECK(hipMemcpy(&res_f, fX, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&res_d, dX, sizeof(double), hipMemcpyDeviceToHost));

    hipFree(dX);
    hipFree(fX);

    HIP_CHECK(hipModuleUnload(module));

    REQUIRE(fabs((res_f/1000) - f_val) <= 0.2f);
    REQUIRE(fabs((res_d/1000) - d_val) <= 0.2);
  }
}
