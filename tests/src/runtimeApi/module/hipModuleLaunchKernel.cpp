/*
   Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */
/* Test Scenarios
  1. hipModuleLaunchKernel Negative Scenarios
 */
/* HIT_START
 * BUILD_CMD: matmul.code %hc --genco %S/matmul.cpp -o matmul.code
 * BUILD: %t %s  ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */


#include "test_common.h"

#define fileName "matmul.code"
#define matmulK "matmulK"
#define SixteenSec "SixteenSecKernel"
#define KernelandExtra "KernelandExtraParams"
#define FourSec "FourSecKernel"
#define TwoSec "TwoSecKernel"


bool Module_Negative_tests() {
  bool testStatus = true;
  HIPCHECK(hipSetDevice(0));
  hipError_t err;
  struct {
    void* _Ad;
    void* _Bd;
    void* _Cd;
    int _n;
  } args1;
  args1._Ad = nullptr;
  args1._Bd = nullptr;
  args1._Cd = nullptr;
  args1._n  = 0;
  hipFunction_t MultKernel, KernelandExtraParamKernel;
  size_t size1;
  size1 = sizeof(args1);
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  hipStream_t stream1;
  hipDeviceptr_t *Ad;
  hipDevice_t device;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipCtxCreate(&context, 0, device));
#endif

  HIPCHECK(hipModuleLoad(&Module, fileName));
  HIPCHECK(hipModuleGetFunction(&MultKernel, Module, matmulK));
  HIPCHECK(hipModuleGetFunction(&KernelandExtraParamKernel,
                                Module, KernelandExtra));
  void *config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
    HIP_LAUNCH_PARAM_END};
  void *params[] = {Ad};
  HIPCHECK(hipStreamCreate(&stream1));
  // Passing nullptr to kernel function
  err = hipModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0,
      stream1, NULL,
      reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed nullptr to kernel function");
    testStatus = false;
  }
  // Passing Max int value to block dimensions
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                              std::numeric_limits<uint32_t>::max(),
                              std::numeric_limits<uint32_t>::max(),
                              std::numeric_limits<uint32_t>::max(),
                              0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for max values to block dimension");
    testStatus = false;
  }
  // Passing both kernel and extra params
  err = hipModuleLaunchKernel(KernelandExtraParamKernel, 1, 1, 1, 1,
                              1, 1, 0, stream1,
                              reinterpret_cast<void**>(&params),
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel fail when we pass both kernel,extra args");
    testStatus = false;
  }
  // Passing more than maxthreadsperblock to block dimensions
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                              deviceProp.maxThreadsPerBlock+1,
                              deviceProp.maxThreadsPerBlock+1,
                              deviceProp.maxThreadsPerBlock+1, 0, stream1, NULL,
                              reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for max group size");
    testStatus = false;
  }
  // Passing invalid config data to extra params
  void *config3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
      reinterpret_cast<void**>(&config3));
  if (err == hipSuccess) {
    printf("hipExtModuleLaunchKernel failed for invalid conf \n");
    testStatus = false;
  }
  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  hipCtxDestroy(context);
#endif
  return testStatus;
}

int main(int argc, char* argv[]) {
  bool testStatus = true;
  testStatus = Module_Negative_tests();
  if (testStatus) {
    passed();
  } else {
    failed("Test Failed!");
  }
}
