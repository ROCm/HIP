/*
   Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
/*
This testfile verifies __builtin_amdgcn_global_atomic_fadd_f64 API scenarios
1. AtomicAdd on Coherent Memory
2. AtomicAdd on Non-Coherent Memory
3. AtomicAdd on Coherent Memory with RTC
4. AtomicAdd on Non-Coherent Memory with RTC
*/

#include<hip_test_checkers.hh>
#include<hip_test_common.hh>
#include <hip/hiprtc.h>

#define INC_VAL 10
#define INITIAL_VAL 5
__global__ void AtomicAdd_GlobalMem(double* addr, double* result) {
  double inc_val = 10;
  *result = __builtin_amdgcn_global_atomic_fadd_f64(addr, inc_val);
}
static constexpr auto AtomicAddGlobalMem{
R"(
extern "C"
__global__ void AtomicAdd_GlobalMem(double* addr, double* result) {
  double inc_val = 10;
  *result = __builtin_amdgcn_global_atomic_fadd_f64(addr, inc_val);
}
)"};
/*
This test verifies the built in atomic add API on Coherent Memory
Input: A_h with INITIAL_VAL
Output: A_h will not get updated with Coherent Memory
        A_h will be INITIAL_VAL
        ret value would be 0, B_h would be 0
*/
TEST_CASE("Unit_BuiltInAtomicAdd_CoherentGlobalMem") {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does support HostPinned Memory");
    } else {
      double *A_h, *result_h, *result;
      double *A_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
            hipHostMallocCoherent));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result_h),
                              sizeof(double), hipHostMallocCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
                                        A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result),
                                        result_h, 0));
       std::cout << "test" << std::endl;
      hipLaunchKernelGGL(AtomicAdd_GlobalMem, dim3(1), dim3(1),
                         0, 0, A_d,
                         result);
       std::cout << "test 1" << std::endl;
      HIP_CHECK(hipDeviceSynchronize());
      REQUIRE(A_h[0] == INITIAL_VAL);
      REQUIRE(*result_h == 0);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipFree(result));
    }
  } else {
    SUCCEED("Memory model feature is only supported for gfx90a Hence"
            "skipping the testcase for this GPU " << device);
  }
}

/*
This test verifies the built in atomic add API on Non-Coherent Memory
Input: A_h with INITIAL_VAL
Output: A_h will not get updated with Coherent Memory
        A_h will be INITIAL_VAL+INC_VAL
        B_h would be initial value of A_h, B_h would be INITIAL_VAL
*/
TEST_CASE("Unit_BuiltInAtomicAdd_NonCoherentGlobalMem") {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      double *A_h, *result, *B_h;
      double *A_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
            hipHostMallocNonCoherent));
      B_h = reinterpret_cast<double*>(malloc(sizeof(double)));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), sizeof(double)));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
                                        A_h, 0));
      hipLaunchKernelGGL(AtomicAdd_GlobalMem, dim3(1), dim3(1),
                         0, 0, static_cast<double* >(A_d),
                         static_cast<double* >(result));
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(B_h, result, sizeof(double), hipMemcpyDeviceToHost));
      REQUIRE(A_h[0] == INITIAL_VAL + INC_VAL);
      REQUIRE(*B_h == INITIAL_VAL);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipFree(result));
      free(B_h);
    }
  } else {
    SUCCEED("Memory model feature is only supported for gfx90a"
            "Hence skipping the testcase for GPU-0");
  }
}
/*
This test verifies the built in atomic add API on Coherent Memory with RTC
Input: A_h with INITIAL_VAL
Output: A_h will not get updated with Coherent Memory
        A_h will be INITIAL_VAL
        ret value would be 0, B_h would be 0
*/
TEST_CASE("Unit_BuiltInAtomicAdd_CoherentGlobalMemWithRtc") {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      hiprtcProgram prog;
      hiprtcCreateProgram(&prog,        // prog
                          AtomicAddGlobalMem,       // buffer
                          "kernel.cu",  // name
                          0, nullptr, nullptr);
      std::string sarg = std::string("--gpu-architecture=") + prop.gcnArchName;
      const char* options[] = {sarg.c_str()};
      hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        INFO(log);
      }

      REQUIRE(compileResult == HIPRTC_SUCCESS);
      size_t codeSize;
      HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

      std::vector<char> code(codeSize);
      HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
      HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

      hipModule_t module;
      hipFunction_t fmaxkernel;
      HIP_CHECK(hipModuleLoadData(&module, code.data()));
      HIP_CHECK(hipModuleGetFunction(&fmaxkernel, module,
                                     "AtomicAdd_GlobalMem"));
      double *A_h, *result, *B_h;
      double *A_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
                              hipHostMallocCoherent));
      B_h = reinterpret_cast<double*>(malloc(sizeof(double)));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), sizeof(double)));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
                                        A_h, 0));
      struct {
        double* p;
        double* res;
      } args_f{A_d, result};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &size, HIP_LAUNCH_PARAM_END};
      hipModuleLaunchKernel(fmaxkernel, 1, 1, 1, 1, 1, 1, 0,
                            nullptr, nullptr, config_d);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(B_h, result, sizeof(double), hipMemcpyDeviceToHost));
      REQUIRE(A_h[0] == INITIAL_VAL);
      REQUIRE(*B_h == 0);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipFree(result));
      free(B_h);
    }
  } else {
    SUCCEED("Memory model feature is only supported for gfx90a, Hence"
             "skipping the testcase for this GPU " << device);
  }
}

/*
This test verifies the built in atomic add API on Non-Coherent Memory
Input: A_h with INITIAL_VAL
Output: A_h will not get updated with Coherent Memory
        A_h will be INITIAL_VAL+INC_VAL
        B_h would be initial value of A_h, B_h would be INITIAL_VAL
*/
TEST_CASE("Unit_BuiltInAtomicAdd_NonCoherentGlobalMemWithRtc") {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does support HostPinned Memory");
    } else {
      hiprtcProgram prog;
      hiprtcCreateProgram(&prog,        // prog
                          AtomicAddGlobalMem,       // buffer
                          "kernel.cu",  // name
                          0, nullptr, nullptr);
      std::string sarg = std::string("--gpu-architecture=") + prop.gcnArchName;
      const char* options[] = {sarg.c_str()};
      hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }

      REQUIRE(compileResult == HIPRTC_SUCCESS);
      size_t codeSize;
      HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

      std::vector<char> code(codeSize);
      HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
      HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

      hipModule_t module;
      hipFunction_t fmaxkernel;
      HIP_CHECK(hipModuleLoadData(&module, code.data()));
      HIP_CHECK(hipModuleGetFunction(&fmaxkernel, module,
                                     "AtomicAdd_GlobalMem"));
      double *A_h, *result, *B_h;
      double *A_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
            hipHostMallocNonCoherent));
      B_h = reinterpret_cast<double*>(malloc(sizeof(double)));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), sizeof(double)));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
                                        A_h, 0));
      struct {
        double* p;
        double* res;
      } args_f{A_d, result};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &size, HIP_LAUNCH_PARAM_END};
      hipModuleLaunchKernel(fmaxkernel, 1, 1, 1, 1, 1, 1, 0,
                            nullptr, nullptr, config_d);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(B_h, result, sizeof(double), hipMemcpyDeviceToHost));
      REQUIRE(A_h[0] == INITIAL_VAL + INC_VAL);
      REQUIRE(*B_h == INITIAL_VAL);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipFree(result));
      free(B_h);
    }
  } else {
    SUCCEED("Memory model feature is only supported for gfx90a, Hence"
             "skipping the testcase for this GPU " << device);
  }
}
