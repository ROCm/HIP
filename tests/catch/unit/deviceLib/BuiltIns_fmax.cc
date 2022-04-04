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
This testfile verifies Built fmax  API scenarios
1. Builtin fmax on Coherent Memory with memory type as global
2. Builtin fmax on Non-Coherent Memory with memory type as global
3. Builtin fmax with memory type as flat
4. Builtin fmax on Coherent Memory with RTC and memory type as global
5. Builtin fmax on Non-Coherent Memory with RTC and memory type as global
6. Builtin fmax with RTC and memory type as flat
*/


#include<hip_test_checkers.hh>
#include<hip_test_common.hh>
#include <hip/hiprtc.h>

#define INITIAL_VAL 5

__global__ void unsafeAtomicMax_FlatMem(double* addr, double* result) {
  __shared__ double int_val;
  int_val = 5;
  double comp = 10;
  if (__builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)(&int_val)))
    *result = __builtin_amdgcn_flat_atomic_fmax_f64(&int_val, comp);
  else
    *result = __builtin_amdgcn_global_atomic_fmax_f64(&int_val, comp);
  *addr = int_val;
}
__global__ void unsafeAtomicMax_GlobalMem(double* addr, double* result) {
  double comp = 10;
  if (__builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)(addr)))
    *result =  __builtin_amdgcn_flat_atomic_fmax_f64(addr, comp);
  else
    *result =  __builtin_amdgcn_global_atomic_fmax_f64(addr, comp);
}
static constexpr auto fmaxFlatMem {
R"(
extern "C"
__global__ void unsafeAtomicMax_FlatMem(double* addr, double* result) {
  __shared__ double int_val;
  int_val = 5;
  double comp = 10;
  if (__builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)(&int_val)))
    *result = __builtin_amdgcn_flat_atomic_fmax_f64(&int_val, comp);
  else
    *result = __builtin_amdgcn_global_atomic_fmax_f64(&int_val, comp);
  *addr = int_val;
}
)"};

static constexpr auto fmaxGlobalMem {
R"(
extern "C"
__global__ void unsafeAtomicMax_GlobalMem(double* addr, double* result) {
  double comp = 10;
  if (__builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)(addr)))
    *result =  __builtin_amdgcn_flat_atomic_fmax_f64(addr, comp);
  else
    *result =  __builtin_amdgcn_global_atomic_fmax_f64(addr, comp);
}
)"};

/*
This testcase verifies the builtinAtomic fmax API on Coherent memory
with memory type as global
Input: A_h with INITIAL_VAL
Output: Return val would be 0 and the input value to API will not
        get updated. A_h would be INITIAL_VAL, B_h is 0
*/
TEST_CASE("Unit_BuiltinAtomics_fmaxCoherentGlobalMem") {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      double *A_h, *B_h;
      double *A_d;
      double *result;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
            hipHostMallocCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      B_h = reinterpret_cast<double*>(malloc(sizeof(double)));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), sizeof(double)));
      hipLaunchKernelGGL(unsafeAtomicMax_GlobalMem, dim3(1), dim3(1),
          0, 0, static_cast<double* >(A_d), result);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(B_h, result, sizeof(double), hipMemcpyDeviceToHost));
      REQUIRE(*B_h == 0);
      REQUIRE(A_h[0] == INITIAL_VAL);
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
This testcase verifies the builtinAtomic fmax API
1. Non Coherent memory with memory type as global
2. Memory type as flat
Input: A_h with INITIAL_VAL
Output: Return val would be initial val of A_h and the input value of
        API would be updated with the max value
        A_h would be 10, B_h would be INITIAL_VAL
*/
TEST_CASE("Unit_BuiltinAtomics_fmaxNonCoherentGlobalFlatMem") {
  int mem_type = GENERATE(0, 1);
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      double *A_h, *B_h;
      double *A_d;
      double *result;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
            hipHostMallocNonCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      B_h = reinterpret_cast<double*>(malloc(sizeof(double)));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), sizeof(double)));
      if (mem_type) {
        hipLaunchKernelGGL(unsafeAtomicMax_GlobalMem, dim3(1), dim3(1),
            0, 0, static_cast<double* >(A_d), result);
      } else {
        hipLaunchKernelGGL(unsafeAtomicMax_FlatMem, dim3(1), dim3(1),
            0, 0, static_cast<double* >(A_d), result);
      }
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(B_h, result, sizeof(double), hipMemcpyDeviceToHost));
      REQUIRE(*B_h == INITIAL_VAL);
      REQUIRE(A_h[0] == 10);
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
This testcase verifies the builtinAtomic fmax API on Coherent memory
with RTC and memory type as global
Input: A_h with INITIAL_VAL
Output: Return val would be 0 and the input value to API will not
        get updated. A_h would be INITIAL_VAL, B_h is 0
*/
TEST_CASE("Unit_BuiltinAtomicsRTC_fmaxCoherentGlobalMem") {
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
          fmaxGlobalMem,       // buffer
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
                                     "unsafeAtomicMax_GlobalMem"));

      double *A_h, *B_h;
      double *A_d;
      double *result;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
            hipHostMallocCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      B_h = reinterpret_cast<double*>(malloc(sizeof(double)));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), sizeof(double)));
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
      REQUIRE(*B_h == 0);
      REQUIRE(A_h[0] == INITIAL_VAL);
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
This testcase verifies the builtinAtomic fmax API with RTC
1. Non Coherent memory with memory type as global
2. Memory type as flat
Input: A_h with INITIAL_VAL
Output: Return val would be initial val of A_h and the input value of
        API would be updated with the max value
        A_h would be 10, B_h would be INITIAL_VAL
*/
TEST_CASE("Unit_BuiltinAtomicsRTC_fmaxNonCoherentGlobalFlatMem") {
  int mem_type = GENERATE(0, 1);
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
      if (mem_type) {
        hiprtcCreateProgram(&prog,        // prog
            fmaxGlobalMem,       // buffer
            "kernel.cu",  // name
            0, nullptr, nullptr);
      } else {
        hiprtcCreateProgram(&prog,        // prog
            fmaxFlatMem,       // buffer
            "kernel.cu",  // name
            0, nullptr, nullptr);
      }
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
      if (mem_type) {
        HIP_CHECK(hipModuleGetFunction(&fmaxkernel, module,
                                       "unsafeAtomicMax_GlobalMem"));
      } else {
        HIP_CHECK(hipModuleGetFunction(&fmaxkernel, module,
                                       "unsafeAtomicMax_FlatMem"));
      }

      double *A_h, *B_h;
      double *A_d;
      double *result;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(double),
            hipHostMallocNonCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      B_h = reinterpret_cast<double*>(malloc(sizeof(double)));
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), sizeof(double)));
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
      REQUIRE(*B_h == INITIAL_VAL);
      REQUIRE(A_h[0] == 10);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipFree(result));
      free(B_h);
    }
  } else {
    SUCCEED("Memory model feature is only supported for gfx90a, Hence"
            "skipping the testcase for this GPU " << device);
  }
}
