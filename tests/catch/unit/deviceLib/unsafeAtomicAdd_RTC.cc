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
unsafeAtomicAdd Scenarios with hipRTC:
1. FineGrainMemory with -m-nounsafe-fp-atomics flag
2. FineGrainMemory without compilation flag
3. FineGrainMemory without -munsafe-fp-atomics flag
4. CoarseGrainMemory with -m-nounsafe-fp-atomics flag
5. CoarseGrainMemory without compilation flag
6. CoarseGrainMemory without -munsafe-fp-atomics flag
*/

#include<hip_test_checkers.hh>
#include<hip_test_common.hh>
#include <hip/hiprtc.h>
#define INCREMENT_VAL 10
#define INITIAL_VAL 5

static constexpr auto fkernel{
R"(
extern "C"
__global__ void AtomicCheck(float* Ad, float *result) {
*result = unsafeAtomicAdd(Ad, 10);
}
)"};

static constexpr auto dkernel{
R"(
extern "C"
__global__ void AtomicCheck(double* Ad, double *result) {
*result = unsafeAtomicAdd(Ad, 10);
}
)"};

/*
   Test unsafeAtomicAdd API for the fine grained memory variable
   where kernel is compiled using hipRTC and with
   compilation flag -mno-unsafe-fp-atomics.
   Input: Ad{5}, INCREMENT_VAL{10}
   Output: unsafeAtomicAdd API will not work and returns 0 so
   the initial value will be intact. expected O/P is 5
*/
TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_CoherentRTCnounsafeatomicflag", "",
                   float, double) {
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    hiprtcProgram prog;
    if (std::is_same<TestType, float>::value) {
    hiprtcCreateProgram(&prog,        // prog
                        fkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    } else {
     hiprtcCreateProgram(&prog,        // prog
                        dkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    }
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
    const char* options[] = {sarg.c_str(), "-mno-unsafe-fp-atomics"};
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, options)};
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
    hipFunction_t f_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "AtomicCheck"));
    if (props.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h, *result;
      TestType *A_d, *result_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                              hipHostMallocCoherent));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result),
                              sizeof(TestType),
                              hipHostMallocCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d),
            result, 0));
      struct {
        TestType* p;
        TestType* result;
      } args_f{A_d, result_d};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size, HIP_LAUNCH_PARAM_END};
      HIP_CHECK(
        hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1, 1, 0,
          nullptr, nullptr, config_d));
      HIP_CHECK(hipDeviceSynchronize());
      REQUIRE(A_h[0] == INITIAL_VAL);
      REQUIRE(*result == 0);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
    HIP_CHECK(hipModuleUnload(module));
  } else {
      SUCCEED("Memory model feature is only supported for gfx90a, Hence"
               "skipping the testcase for this GPU " << device);
  }
}


/*
   Test unsafeAtomicAdd API for the fine grained memory variable
   where kernel is compiled using hipRTC and with
   compilation flag -munsafe-fp-atomics.
   Input: Ad{5}, INCREMENT_VAL{10}
   Output: unsafeAtomicAdd API will not work and r`eturns 0 so
   the initial value will be intact. expected O/P is 5
*/
TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_CoherentRTCunsafeatomicflag", "",
                   float, double) {
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    hiprtcProgram prog;
    if (std::is_same<TestType, float>::value) {
    hiprtcCreateProgram(&prog,        // prog
                        fkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    } else {
     hiprtcCreateProgram(&prog,        // prog
                        dkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    }
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
    const char* options[] = {sarg.c_str(), "-munsafe-fp-atomics"};
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, options)};

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
    hipFunction_t f_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "AtomicCheck"));

    if (props.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h, *result;
      TestType *A_d, *result_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                              hipHostMallocCoherent));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result),
                              sizeof(TestType),
                              hipHostMallocCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d),
            result, 0));
      struct {
        TestType* p;
        TestType* result;
      } args_f{A_d, result_d};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size, HIP_LAUNCH_PARAM_END};
      HIP_CHECK(
        hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1, 1, 0,
          nullptr, nullptr, config_d));
      HIP_CHECK(hipDeviceSynchronize());
      REQUIRE(A_h[0] == INITIAL_VAL);
      REQUIRE(*result == 0);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
    HIP_CHECK(hipModuleUnload(module));
  } else {
      SUCCEED("Memory model feature is only supported for gfx90a, Hence"
               "skipping the testcase for this GPU " << device);
  }
}

/* Test unsafeAtomicAdd API for the fine grained memory variable
   where kernel is compiled using hipRTC and without  compilation flag
   Input: Ad{5}, INCREMENT_VAL{10}
   Output: unsafeAtomicAdd API will not work and returns 0 so
   the initial value will be intact. expected O/P is 5*/

TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_CoherentRTCwithoutflag", "",
                   float, double) {
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
     hiprtcProgram prog;
    if (std::is_same<TestType, float>::value) {
    hiprtcCreateProgram(&prog,        // prog
                        fkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    } else {
     hiprtcCreateProgram(&prog,        // prog
                        dkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    }
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
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
    hipFunction_t f_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "AtomicCheck"));

    if (props.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h, *result;
      TestType *A_d, *result_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(float),
                              hipHostMallocCoherent));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result), sizeof(float),
                              hipHostMallocCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
                                        A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d),
                                        result, 0));
      struct {
        TestType* p;
        TestType* result;
      } args_f{A_d, result_d};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &size, HIP_LAUNCH_PARAM_END};
      HIP_CHECK(
        hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1,
                            1, 0, nullptr, nullptr, config_d));
      HIP_CHECK(hipDeviceSynchronize());
      REQUIRE(A_h[0] == INITIAL_VAL);
      REQUIRE(*result == 0);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
    HIP_CHECK(hipModuleUnload(module));
  } else {
      SUCCEED("Memory model feature is only supported for gfx90a, Hence"
              "skipping the testcase for this GPU " << device);
  }
}

/*
   Test unsafeAtomicAdd API for the coarse grained memory variable where kernel
   is compiled using hipRTC and with compilation flag -mno-unsafe-fp-atomics
   Input: Ad{5}, INCREMENT_VAL{10}
   Output: Expected O/P is 15 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_NonCoherentRTCnounsafeatomicflag", "",
                   float, double) {
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
     hiprtcProgram prog;
    if (std::is_same<TestType, float>::value) {
    hiprtcCreateProgram(&prog,        // prog
                        fkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    } else {
      hiprtcCreateProgram(&prog,        // prog
                        dkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    }
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
    const char* options[] = {sarg.c_str(), "-mno-unsafe-fp-atomics"};
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, options)};

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
    hipFunction_t f_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "AtomicCheck"));
    if (props.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h, *result;
      TestType *A_d, *result_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                              hipHostMallocNonCoherent));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result),
                              sizeof(TestType)));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d),
            result, 0));
      struct {
        TestType* p;
        TestType* result;
      } args_f{A_d, result_d};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size, HIP_LAUNCH_PARAM_END};
      HIP_CHECK(
        hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1, 1, 0,
                            nullptr, nullptr, config_d));
      HIP_CHECK(hipDeviceSynchronize());
      REQUIRE(A_h[0] == INITIAL_VAL + INCREMENT_VAL);
      REQUIRE(*result == INITIAL_VAL);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
    HIP_CHECK(hipModuleUnload(module));
  } else {
      SUCCEED("Memory model feature is only supported for gfx90a, Hence"
              "skipping the testcase for this GPU " << device);
  }
}

/*
   Test unsafeAtomicAdd API for the coarse grained memory variable where kernel
   is compiled using hipRTC and with compilation flag -munsafe-fp-atomics
   Input: Ad{5}, INCREMENT_VAL{10}
   Output: Expected O/P is 15 */

TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_NonCoherentRTCunsafeatomicflag", "",
                   float, double) {
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
     hiprtcProgram prog;
    if (std::is_same<TestType, float>::value) {
    hiprtcCreateProgram(&prog,        // prog
                        fkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    } else {
     hiprtcCreateProgram(&prog,        // prog
                        dkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    }
    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
    const char* options[] = {sarg.c_str(), "-munsafe-fp-atomics"};
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, options)};

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
    hipFunction_t f_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "AtomicCheck"));

    if (props.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h, *result;
      TestType *A_d, *result_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                              hipHostMallocNonCoherent));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result),
                              sizeof(TestType)));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d),
            result, 0));
      struct {
        TestType* p;
        TestType* result;
      } args_f{A_d, result_d};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size, HIP_LAUNCH_PARAM_END};
      HIP_CHECK(
        hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1, 1, 0,
                            nullptr, nullptr, config_d));
      HIP_CHECK(hipDeviceSynchronize());
      REQUIRE(A_h[0] == INITIAL_VAL + INCREMENT_VAL);
      REQUIRE(*result == INITIAL_VAL);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
    HIP_CHECK(hipModuleUnload(module));
  } else {
      SUCCEED("Memory model feature is only supported for gfx90a, Hence"
              "skipping the testcase for this GPU " << device);
  }
}

/*
   Test unsafeAtomicAdd API for the coarse  grained memory variable
   where kernel is compiled using hipRTC and without  compilation flag
   Input: Ad{5}, INCREMENT_VAL{10}
   Output: O/P is 15 */

TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_NonCoherentRTC", "",
                   float, double) {
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    hiprtcProgram prog;
    if (std::is_same<TestType, float>::value) {
    hiprtcCreateProgram(&prog,        // prog
                        fkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    } else {
     hiprtcCreateProgram(&prog,        // prog
                        dkernel,       // buffer
                        "kernel.cu",  // name
                        0, nullptr, nullptr);
    }

    std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
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
    hipFunction_t f_kernel;
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    HIP_CHECK(hipModuleGetFunction(&f_kernel, module, "AtomicCheck"));

    if (props.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h, *result;
      TestType *A_d, *result_d;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                              hipHostMallocNonCoherent));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result),
                              sizeof(TestType)));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
            A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d),
           result, 0));
      struct {
        TestType* p;
        TestType* result;
      } args_f{A_d, result_d};
      auto size = sizeof(args_f);
      void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &size, HIP_LAUNCH_PARAM_END};
      HIP_CHECK(
        hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1, 1, 0,
                            nullptr, nullptr, config_d));
      HIP_CHECK(hipDeviceSynchronize());
      REQUIRE(A_h[0] == INITIAL_VAL + INCREMENT_VAL);
      REQUIRE(*result == INITIAL_VAL);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
    HIP_CHECK(hipModuleUnload(module));
  } else {
      SUCCEED("Memory model feature is only supported for gfx90a, Hence"
              "skipping the testcase for this GPU " << device);
  }
}
