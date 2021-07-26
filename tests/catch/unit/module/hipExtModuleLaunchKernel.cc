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
  2. hipExtModuleLaunchKernel API verifying the kernel execution time of a particular kernel.
  3. hipExtModuleLaunchKernel API verifying the kernel execution time by disabling the time flag
  4. hipModuleLaunchKernel Work Group tests =>
     - (block.x * block.y * block.z) <= Work Group Size
       where block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
     - (block.x * block.y * block.z) > Work Group Size
       where block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
 */

#include <math.h>
#include "hip_test_common.hh"
#include "hip_test_kernels.hh"
#include "hip/hip_ext.h"

#define fileName "module_kernels.code"
#define matmulK "matmulK"
#define SixteenSec "SixteenSecKernel"
#define KernelandExtra "KernelandExtraParams"
#define FourSec "FourSecKernel"
#define TwoSec "TwoSecKernel"
#define globalDevVar "deviceGlobal"
#define dummyKernel "EmptyKernel"
#define FOURSEC_KERNEL 4999
#define TWOSEC_KERNEL  2999

struct gridblockDim {
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int blockZ;
};

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
  TwoSecKernel, KernelandExtraParamKernel, DummyKernel;
  struct {
    int clockRate;
    void* _Ad;
    void* _Bd;
    void* _Cd;
    int _n;
  } args1, args2;
  struct {
  } args3;
  size_t size1;
  size_t size2;
  size_t size3;
  size_t deviceGlobalSize;
 public :
  void AllocateMemory();
  void DeAllocateMemory();
  void ModuleLoad();
  void Module_Negative_tests();
  void ExtModule_Negative_tests();
  void Module_WorkGroup_Test();
  void ExtModule_KernelExecutionTime();
  void ExtModule_Disabled_Timingflag();
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
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad),
           SIZE*sizeof(int)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd),
           SIZE*sizeof(int)));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&C), SIZE*sizeof(int)));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE*sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE*sizeof(int), hipMemcpyHostToDevice));
  int clkRate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
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
  size3 = sizeof(args3);
  HIP_CHECK(hipEventCreate(&start_event1));
  HIP_CHECK(hipEventCreate(&end_event1));
  HIP_CHECK(hipEventCreate(&start_event2));
  HIP_CHECK(hipEventCreate(&end_event2));
  HIP_CHECK(hipEventCreateWithFlags(&start_timingDisabled,
                                   hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&end_timingDisabled,
                                   hipEventDisableTiming));
}

void ModuleLaunchKernel::ModuleLoad() {
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&MultKernel, Module, matmulK));
  HIP_CHECK(hipModuleGetFunction(&SixteenSecKernel, Module, SixteenSec));
  HIP_CHECK(hipModuleGetFunction(&KernelandExtraParamKernel,
                                Module, KernelandExtra));
  HIP_CHECK(hipModuleGetFunction(&FourSecKernel, Module, FourSec));
  HIP_CHECK(hipModuleGetFunction(&TwoSecKernel, Module, TwoSec));
  HIP_CHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize,
                              Module, globalDevVar));
}

void ModuleLaunchKernel::DeAllocateMemory() {
  HIP_CHECK(hipEventDestroy(start_event1));
  HIP_CHECK(hipEventDestroy(end_event1));
  HIP_CHECK(hipEventDestroy(start_event2));
  HIP_CHECK(hipEventDestroy(end_event2));
  HIP_CHECK(hipEventDestroy(start_timingDisabled));
  HIP_CHECK(hipEventDestroy(end_timingDisabled));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipHostFree(C));
  HIP_CHECK(hipModuleUnload(Module));
}
/*
 * In this scenario,We launch the 4 sec kernel and 2 sec kernel
 * and we fetch the event execution time of each kernel and it
 * should not exceed the execution time of that particular kernel
 */
void ModuleLaunchKernel::ExtModule_KernelExecutionTime() {
  HIP_CHECK(hipSetDevice(0));
  AllocateMemory();
  ModuleLoad();
  float time_4sec, time_2sec;
  void *config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};

  // Launching kernels
  HIP_CHECK(hipExtModuleLaunchKernel(FourSecKernel, 1, 1, 1, 1, 1, 1, 0,
                                     stream1,
                                     NULL, reinterpret_cast<void**>(&config2),
                                     start_event1, end_event1, 0));
  HIP_CHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                     NULL, reinterpret_cast<void**>(&config2),
                                     start_event2, end_event2, 0));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipEventElapsedTime(&time_4sec, start_event1, end_event1));
  HIP_CHECK(hipEventElapsedTime(&time_2sec, start_event2, end_event2));

  INFO("Expected Vs Actual: Kernel1-<" <<  FOURSEC_KERNEL << "Vs" << time_4sec
       << "Kernel2-<" << TWOSEC_KERNEL << "Vs" << time_2sec);
  // Verifying the kernel execution time
  REQUIRE(time_4sec < static_cast<float>(FOURSEC_KERNEL));
  REQUIRE(time_2sec < static_cast<float>(TWOSEC_KERNEL));

  DeAllocateMemory();
}
/*
 * In this Scenario, we create events by disabling the timing flag
 * We then Launch the kernel using hipExtModuleLaunchKernel by passing
 * disabled events and try to fetch kernel execution time using
 * hipEventElapsedTime API which would fail as the flag is disabled.
 */
void ModuleLaunchKernel::ExtModule_Disabled_Timingflag() {
  // Allocating Memory and Loading kernel
  AllocateMemory();
  ModuleLoad();
  float time_2sec;
  void *config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
    HIP_LAUNCH_PARAM_END};

  // Launching Kernel
  HIP_CHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                    NULL, reinterpret_cast<void**>(&config2),
                                    start_timingDisabled,
                                    end_timingDisabled, 0));
  HIP_CHECK(hipStreamSynchronize(stream1));

  REQUIRE(hipEventElapsedTime(&time_2sec, start_timingDisabled,
          end_timingDisabled) != hipSuccess);

  // DeAllocating the memory
  DeAllocateMemory();
}

/*
This testcase verifies negative scenarios of hipExtModuleLaunchKernel API
*/
void ModuleLaunchKernel::ExtModule_Negative_tests() {
  HIP_CHECK(hipSetDevice(0));
  // Allocating memeory and loading kernel
  AllocateMemory();
  ModuleLoad();
  void *config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  void *params[] = {Ad};

  SECTION("Nullptr to kernel function") {
    REQUIRE(hipExtModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0,
                                     stream1, NULL,
                                     reinterpret_cast<void**>(&config1),
                                      nullptr, nullptr, 0) != hipSuccess);
  }

  SECTION("Max int value to block dimensions") {
    REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 1, 1,
                                     std::numeric_limits<uint32_t>::max(),
                                     std::numeric_limits<uint32_t>::max(),
                                     std::numeric_limits<uint32_t>::max(), 0,
                                     stream1, NULL,
                                     reinterpret_cast<void**>(&config1),
                                     nullptr, nullptr, 0) != hipSuccess);
  }

    SECTION("Null values to all dimensions") {
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 0, 0, 0,
                                       0,
                                       0,
                                       0, 0,
                                       stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Passing 0 for x dimension") {
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 0, 1, 1,
                                       0,
                                       1,
                                       1, 0,
                                       stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Passing 0 for y dimension") {
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 0, 1,
                                       1,
                                       0,
                                       1, 0,
                                       stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Passing 0 for Z dimension") {
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 1, 0,
                                       1,
                                       1,
                                       0, 0,
                                       stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Passing both kernel and extra params") {
      REQUIRE(hipExtModuleLaunchKernel(KernelandExtraParamKernel, 1, 1, 1, 1,
                                       1, 1, 0,
                                       stream1,
                                       reinterpret_cast<void**>(&params),
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Passing both than maxthreadsperblock to block dimensions") {
      hipDeviceProp_t deviceProp;
      hipGetDeviceProperties(&deviceProp, 0);
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 1, 1,
                                       deviceProp.maxThreadsPerBlock+1,
                                       deviceProp.maxThreadsPerBlock+1,
                                       deviceProp.maxThreadsPerBlock+1, 0,
                                       stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Block dimension x = Max alloweed + 1") {
      hipDeviceProp_t deviceProp;
      hipGetDeviceProperties(&deviceProp, 0);
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 1, 1,
                                       deviceProp.maxThreadsDim[0]+1,
                                       1,
                                       1, 0, stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Block dimension Y = Max alloweed + 1") {
      hipDeviceProp_t deviceProp;
      hipGetDeviceProperties(&deviceProp, 0);
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 1, 1,
                                       1,
                                       deviceProp.maxThreadsDim[1]+1,
                                       1, 0, stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Block dimension Z = Max alloweed + 1") {
      hipDeviceProp_t deviceProp;
      hipGetDeviceProperties(&deviceProp, 0);
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 1, 1,
                                       1,
                                       1,
                                       deviceProp.maxThreadsDim[2]+1, 0,
                                       stream1, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    SECTION("Passing invalid config data in extra params") {
      void *config3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                         HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                         HIP_LAUNCH_PARAM_END};
      REQUIRE(hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, 1, 1, 1, 0,
                                       stream1, NULL,
                                       reinterpret_cast<void**>(&config3),
                                       nullptr, nullptr, 0) != hipSuccess);
    }

    DeAllocateMemory();
}

void ModuleLaunchKernel::Module_WorkGroup_Test() {
  // Allocate memory and load modules
  AllocateMemory();
  ModuleLoad();
  void *config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args3,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size3,
                     HIP_LAUNCH_PARAM_END};
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  double cuberootVal =
                cbrt(static_cast<double>(deviceProp.maxThreadsPerBlock));
  uint32_t cuberoot_floor = floor(cuberootVal);
  uint32_t cuberoot_ceil = ceil(cuberootVal);

  // Scenario: (block.x * block.y * block.z) <= Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  HIP_CHECK(hipExtModuleLaunchKernel(DummyKernel,
                            1, 1, 1,
                            cuberoot_floor, cuberoot_floor, cuberoot_floor,
                            0, stream1, NULL,
                            reinterpret_cast<void**>(&config1),
                            nullptr, nullptr, 0));

  // Scenario: (block.x * block.y * block.z) > Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  REQUIRE(hipExtModuleLaunchKernel(DummyKernel,
                            1, 1, 1,
                            cuberoot_ceil, cuberoot_ceil, cuberoot_ceil+1,
                            0, stream1, NULL,
                            reinterpret_cast<void**>(&config1),
                            nullptr, nullptr, 0) != hipSuccess);

  // DeAllocating memory
  DeAllocateMemory();
}

/*
This testcase verifies the negative scenarios of
hipExtModuleLaunchKernel API
*/
TEST_CASE("Unit_hipExtModuleLaunchKernel_Negative") {
  ModuleLaunchKernel Ext_obj;
  Ext_obj.ExtModule_Negative_tests();
}

/*
This testcase verifies hipExtModuleLaunchKernel API by
disabling the timing flag
*/
TEST_CASE("Unit_hipExtModuleLaunchKernel_TimingflagDisabled") {
  ModuleLaunchKernel Ext_obj;
  Ext_obj.ExtModule_Disabled_Timingflag();
}

/*
This testcase verifies hipExtModuleLaunchKernel API kernel
execution time
*/
TEST_CASE("Unit_hipExtModuleLaunchKernel_KernelExecutionTime") {
  ModuleLaunchKernel Ext_obj;
  Ext_obj.ExtModule_KernelExecutionTime();
}

/*
This testcase verifies workgroup of hipExtModuleLaunchKernel API
*/
TEST_CASE("Unit_hipExtModuleLaunchKernel_WorkGroup") {
  ModuleLaunchKernel Ext_obj;
  Ext_obj.Module_WorkGroup_Test();
}
