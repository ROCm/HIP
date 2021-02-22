/*
   Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
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
  1. hipExtModuleLaunchKernel Negative Scenarios
  2. hipExtModuleLaunchKernel concurrency verification using global variable
  3. hipExtModuleLaunchKernel concurrency verification by launching multiple kernels with and
     without concurrency flag and verify the time difference between them
  4. hipExtModuleLaunchKernel API verifying the kernel execution time of a particular kernel.
  5. hipExtModuleLaunchKernel API verifying the kernel execution time by disabling the time flag

  Scenarios 2 and 3 concurrency verification scenarios are not included in HIT command
  as firmware currently does not support the concurrency in the same stream based on the flag.

 */
/* HIT_START
 * BUILD: %t %s  ../../test_common.cpp NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipExtModuleLaunchKernel_NegativeTests --tests 1 EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipExtModuleLaunchKernel_KernelExecutionTime --tests 4 EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipExtModuleLaunchKernel_DisabledEventTimeFlag --tests 5 EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */

#include<chrono>
#include "test_common.h"
#include "hip/hip_ext.h"

#define fileName "matmul.code"
#define matmulK "matmulK"
#define SixteenSec "SixteenSecKernel"
#define KernelandExtra "KernelandExtraParams"
#define FourSec "FourSecKernel"
#define TwoSec "TwoSecKernel"
#define globalDevVar "deviceGlobal"
#define FOURSEC_KERNEL 4999
#define TWOSEC_KERNEL  2999

class ModuleLaunchKernel {
  int N = 64;
  int SIZE = N*N;
  int *A, *B, *C;
  hipDeviceptr_t *Ad, *Bd;
  hipStream_t stream1, stream2;
  hipEvent_t  start_event1, end_event1, start_event2, end_event2,
              start_timingDisabled, end_timingDisabled;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  hipFunction_t MultKernel, SixteenSecKernel, FourSecKernel,
  TwoSecKernel, KernelandExtraParamKernel;
  struct {
    int clockRate;
    void* _Ad;
    void* _Bd;
    void* _Cd;
    int _n;
  } args1, args2;
  size_t size1;
  size_t size2;
  size_t deviceGlobalSize;
 public :
  void AllocateMemory();
  void DeAllocateMemory();
  void ModuleLoad();
  bool Module_Negative_tests();
  bool ExtModule_Negative_tests();
  bool ExtModule_KernelExecutionTime();
  bool ExtModule_ConcurencyCheck_GlobalVar(int conc_flag);
  bool ExtModule_ConcurrencyCheck_TimeVer();
  bool ExtModule_Disabled_Timingflag();
};

void ModuleLaunchKernel::AllocateMemory() {
  A = new int[N*N*sizeof(int)];
  B = new int[N*N*sizeof(int)];
  for (int i=0; i < N; i++) {
    for (int j=0; j < N; j++) {
      A[i*N +j] = 1;
      B[i*N +j] = 1;
    }
  }
  HIPCHECK(hipStreamCreate(&stream1));
  HIPCHECK(hipStreamCreate(&stream2));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Ad),
           SIZE*sizeof(int)));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Bd),
           SIZE*sizeof(int)));
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&C), SIZE*sizeof(int)));
  HIPCHECK(hipMemcpy(Ad, A, SIZE*sizeof(int), hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Bd, B, SIZE*sizeof(int), hipMemcpyHostToDevice));
  int clkRate = 0;
  HIPCHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
  args1._Ad = Ad;
  args1._Bd = Bd;
  args1._Cd = C;
  args1._n  = N;
  args1.clockRate = clkRate;
  args2._Ad = NULL;
  args2._Bd = NULL;
  args2._Cd = NULL;
  args2._n  = 0;
  args2.clockRate = clkRate;
  size1 = sizeof(args1);
  size2 = sizeof(args2);
  HIPCHECK(hipEventCreate(&start_event1));
  HIPCHECK(hipEventCreate(&end_event1));
  HIPCHECK(hipEventCreate(&start_event2));
  HIPCHECK(hipEventCreate(&end_event2));
  HIPCHECK(hipEventCreateWithFlags(&start_timingDisabled,
                                   hipEventDisableTiming));
  HIPCHECK(hipEventCreateWithFlags(&end_timingDisabled,
                                   hipEventDisableTiming));
}

void ModuleLaunchKernel::ModuleLoad() {
  HIPCHECK(hipModuleLoad(&Module, fileName));
  HIPCHECK(hipModuleGetFunction(&MultKernel, Module, matmulK));
  HIPCHECK(hipModuleGetFunction(&SixteenSecKernel, Module, SixteenSec));
  HIPCHECK(hipModuleGetFunction(&KernelandExtraParamKernel,
                                Module, KernelandExtra));
  HIPCHECK(hipModuleGetFunction(&FourSecKernel, Module, FourSec));
  HIPCHECK(hipModuleGetFunction(&TwoSecKernel, Module, TwoSec));
  HIPCHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize,
                              Module, globalDevVar));
}

void ModuleLaunchKernel::DeAllocateMemory() {
  HIPCHECK(hipEventDestroy(start_event1));
  HIPCHECK(hipEventDestroy(end_event1));
  HIPCHECK(hipEventDestroy(start_event2));
  HIPCHECK(hipEventDestroy(end_event2));
  HIPCHECK(hipEventDestroy(start_timingDisabled));
  HIPCHECK(hipEventDestroy(end_timingDisabled));
  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipStreamDestroy(stream2));
  delete[] A;
  delete[] B;
  HIPCHECK(hipFree(Ad));
  HIPCHECK(hipFree(Bd));
  HIPCHECK(hipHostFree(C));
  HIPCHECK(hipModuleUnload(Module));
}
/*
 * In this scenario,We launch the 4 sec kernel and 2 sec kernel
 * and we fetch the event execution time of each kernel and it
 * should not exceed the execution time of that particular kernel
 */
bool ModuleLaunchKernel::ExtModule_KernelExecutionTime() {
  bool testStatus = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  ModuleLoad();
  hipError_t e;
  float time_4sec, time_2sec;

  void *config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  HIPCHECK(hipExtModuleLaunchKernel(FourSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                    NULL, reinterpret_cast<void**>(&config2),
                                    start_event1, end_event1, 0));
  HIPCHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                    NULL, reinterpret_cast<void**>(&config2),
                                    start_event2, end_event2, 0));
  HIPCHECK(hipStreamSynchronize(stream1));
  e = hipEventElapsedTime(&time_4sec, start_event1, end_event1);
  e = hipEventElapsedTime(&time_2sec, start_event2, end_event2);
  if (time_4sec < FOURSEC_KERNEL && time_2sec < TWOSEC_KERNEL) {
    testStatus = true;
  } else {
    printf("Expected Vs Actual: Kernel1-<%d Vs %f Kernel2-<%d Vs %f\n",
           FOURSEC_KERNEL, time_4sec, TWOSEC_KERNEL, time_2sec);
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
/*
 * In this Scenario, we create events by disabling the timing flag
 * We then Launch the kernel using hipExtModuleLaunchKernel by passing
 * disabled events and try to fetch kernel execution time using
 * hipEventElapsedTime API which would fail as the flag is disabled.
 */
bool ModuleLaunchKernel::ExtModule_Disabled_Timingflag() {
  bool testStatus = true;
  AllocateMemory();
  ModuleLoad();
  hipError_t e;
  float time_2sec;
  void *config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
    HIP_LAUNCH_PARAM_END};
  HIPCHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                  NULL, reinterpret_cast<void**>(&config2),
                                  start_timingDisabled, end_timingDisabled, 0));
  HIPCHECK(hipStreamSynchronize(stream1));
  e = hipEventElapsedTime(&time_2sec, start_timingDisabled, end_timingDisabled);
  if (e == hipErrorInvalidHandle) {
    testStatus = true;
  } else {
    printf("Event elapsed time is success when time flag is disabled \n");
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
/*
 * In this scenario , we initially create a global device variable in matmul.cpp
 * with initial value as 1 We then launch the four sec and two sec kernels and
 * try to modify the variable.
 * In case of concurrency,the variable gets updated in four sec kernel to 0x2222
 * and then the two sec kernel would be launched parallely which would again
 * modify the global variable to 0x3333
 * In case of non concurrency,the variale gets updated in four sec kernel
 * and then in two sec kernel and the value of global variable would be 0x5555
 */
bool ModuleLaunchKernel::ExtModule_ConcurencyCheck_GlobalVar(int conc_flag) {
  bool testStatus = true;
  int deviceGlobal_h = 0;
  AllocateMemory();
  ModuleLoad();
  void *config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  HIPCHECK(hipExtModuleLaunchKernel(FourSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                    NULL, reinterpret_cast<void**>(&config2),
                                    start_event1, end_event1, conc_flag));
  HIPCHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                    NULL, reinterpret_cast<void**>(&config2),
                                    start_event2, end_event2, conc_flag));
  HIPCHECK(hipStreamSynchronize(stream1));
  HIPCHECK(hipMemcpyDtoH(&deviceGlobal_h, hipDeviceptr_t(deviceGlobal),
                         deviceGlobalSize));
  if (conc_flag && deviceGlobal_h != 0x5555) {
    testStatus = true;
  } else if (!conc_flag && deviceGlobal_h == 0x5555) {
    testStatus = true;
  } else {
    printf("concurrency failed when concurrency flag is %d and global is %x",
           conc_flag, deviceGlobal_h);
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
/* In this scenario,we initially launch 2 kernels,one is sixteen sec kernel
 * and other is matrix multiplication with non-concurrency (flag 0)
 * and we launch the same 2 kernels with concurrency flag 1. We then compare
 * the time difference between the concurrency and non currency kernels.
 * The concurrency kernel duration should be less than the non concurrency
 * duration kernels
 */
bool ModuleLaunchKernel::ExtModule_ConcurrencyCheck_TimeVer() {
  bool testStatus = true;
  AllocateMemory();
  ModuleLoad();
  int mismatch = 0;
  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  auto start = std::chrono::high_resolution_clock::now();
  HIPCHECK(hipExtModuleLaunchKernel(SixteenSecKernel, 1, 1, 1, 1, 1, 1, 0,
                                    stream1, NULL,
                                    reinterpret_cast<void**>(&config2),
                                    NULL, NULL, 0));
  HIPCHECK(hipExtModuleLaunchKernel(MultKernel, N, N, 1, 32, 32 , 1, 0,
                                    stream1, NULL,
                                    reinterpret_cast<void**>(&config1),
                                    NULL, NULL, 0));
  HIPCHECK(hipStreamSynchronize(stream1));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>
                   (stop-start);
  start = std::chrono::high_resolution_clock::now();
  HIPCHECK(hipExtModuleLaunchKernel(SixteenSecKernel, 1, 1, 1, 1, 1, 1, 0,
                                    stream1, NULL,
                                    reinterpret_cast<void**>(&config2),
                                    NULL, NULL, 1));
  HIPCHECK(hipExtModuleLaunchKernel(MultKernel, N, N, 1, 32, 32, 1, 0,
                                    stream1, NULL,
                                    reinterpret_cast<void**>(&config1),
                                    NULL, NULL, 1));
  HIPCHECK(hipStreamSynchronize(stream1));
  stop = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>
                   (stop-start);
  if (!(duration2.count() < duration1.count())) {
    std::cout << "Test failed as there was no time gain observed when"
              << " two kernels were launched using hipExtModuleLaunchKernel()"
              << " with flag 1." <<std::endl;
    testStatus = false;
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (C[i*N + j] != N)
        mismatch++;
    }
  }
  if (mismatch) {
    std::cout << "Test failed as the result of matrix multiplication"
              << "was found incorrect." << std::endl;
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
bool ModuleLaunchKernel::ExtModule_Negative_tests() {
  bool testStatus = true;
  HIPCHECK(hipSetDevice(0));
  hipError_t err;
  AllocateMemory();
  ModuleLoad();
  void *config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  void *config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  void *params[] = {Ad};
  // Passing nullptr to kernel function in hipExtModuleLaunchKernel API
  err = hipExtModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0,
                                 stream1, NULL,
                                 reinterpret_cast<void**>(&config1),
                                 nullptr, nullptr, 0);
  if (err == hipSuccess) {
    printf("hipExtModuleLaunchKernel failed nullptr to kernel function");
    testStatus = false;
  }
  // Passing Max int value to block dimensions
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1,
                                 std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max(), 0,
                                 stream1, NULL,
                                 reinterpret_cast<void**>(&config1),
                                 nullptr, nullptr, 0);
  if (err == hipSuccess) {
    printf("hipExtModuleLaunchKernel failed for max values to block dimension");
    testStatus = false;
  }
  // Passing both kernel and extra params
  err = hipExtModuleLaunchKernel(KernelandExtraParamKernel, 1, 1, 1, 1, 1, 1, 0,
                                 stream1, reinterpret_cast<void**>(&params),
                                 reinterpret_cast<void**>(&config1),
                                 nullptr, nullptr, 0);
  if (err == hipSuccess) {
    printf("hipExtModuleLaunchKernel fail when we pass both kernel,extra args");
    testStatus = false;
  }
  // Passing more than maxthreadsperblock to block dimensions
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1,
                                 deviceProp.maxThreadsPerBlock+1,
                                 deviceProp.maxThreadsPerBlock+1,
                                 deviceProp.maxThreadsPerBlock+1, 0,
                                 stream1, NULL,
                                 reinterpret_cast<void**>(&config1),
                                 nullptr, nullptr, 0);
  if (err == hipSuccess) {
    printf("hipExtModuleLaunchKernel failed for max group size");
    testStatus = false;
  }
  // Passing invalid config data in extra params
  void *config3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config3),
                                 nullptr, nullptr, 0);
  if (err == hipSuccess) {
    printf("hipExtModuleLaunchKernel failed for invalid conf \n");
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}

int main(int argc, char* argv[]) {
  bool testStatus = true;
  HipTest::parseStandardArguments(argc, argv, false);
  ModuleLaunchKernel kernelLaunch;
  if (p_tests == 1) {
    testStatus &= kernelLaunch.ExtModule_Negative_tests();
  } else if (p_tests == 2) {
    testStatus &= kernelLaunch.ExtModule_ConcurencyCheck_GlobalVar(1);
    testStatus &= kernelLaunch.ExtModule_ConcurencyCheck_GlobalVar(0);
  } else if (p_tests == 3) {
    testStatus &= kernelLaunch.ExtModule_ConcurrencyCheck_TimeVer();
  } else if (p_tests == 4) {
    testStatus &= kernelLaunch.ExtModule_KernelExecutionTime();
  } else if (p_tests == 5) {
    testStatus &= kernelLaunch.ExtModule_Disabled_Timingflag();
  } else {
    failed("Didnt receive any valid option.\n");
  }
  if (testStatus) {
    passed();
  } else {
    failed("Test Failed!");
  }
}
