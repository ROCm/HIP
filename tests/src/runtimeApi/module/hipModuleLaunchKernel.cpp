/*
   Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
  2. hipModuleLaunchKernel Corner Scenarios for Grid and Block dimensions
  3. hipModuleLaunchKernel Work Group tests =>
     - (block.x * block.y * block.z) <= Work Group Size
       where block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
     - (block.x * block.y * block.z) > Work Group Size
       where block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
 */
/* HIT_START
 * BUILD_CMD: matmul.code %hc --genco %S/matmul.cpp -o matmul.code
 * BUILD: %t %s  ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2
 * TEST: %t --tests 0x3
 * HIT_END
 */

#include <math.h>
#include "test_common.h"

#define fileName "matmul.code"
#define matmulK "matmulK"
#define SixteenSec "SixteenSecKernel"
#define KernelandExtra "KernelandExtraParams"
#define FourSec "FourSecKernel"
#define TwoSec "TwoSecKernel"
#define dummyKernel "dummyKernel"

struct gridblockDim {
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int blockZ;
};

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
  // Passing 0 as value for all dimensions
  err = hipModuleLaunchKernel(MultKernel, 0, 0, 0,
                                 0,
                                 0,
                                 0, 0,
                                 stream1, NULL,
                                 reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for 0 as value for all dimensions");
    testStatus = false;
  }
  // Passing 0 as value for x dimension
  err = hipModuleLaunchKernel(MultKernel, 0, 1, 1,
                                 0,
                                 1,
                                 1, 0,
                                 stream1, NULL,
                                 reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for 0 as value for x dimension");
    testStatus = false;
  }
  // Passing 0 as value for y dimension
  err = hipModuleLaunchKernel(MultKernel, 1, 0, 1,
                                 1,
                                 0,
                                 1, 0,
                                 stream1, NULL,
                                 reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for 0 as value for y dimension");
    testStatus = false;
  }
  // Passing 0 as value for z dimension
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 0,
                                 1,
                                 1,
                                 0, 0,
                                 stream1, NULL,
                                 reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for 0 as value for z dimension");
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
  // Block dimension X = Max Allowed + 1
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                            deviceProp.maxThreadsDim[0]+1,
                            1,
                            1, 0, stream1, NULL,
                            reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for (MaxBlockDimX + 1)");
    testStatus = false;
  }
  // Block dimension Y = Max Allowed + 1
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                            1,
                            deviceProp.maxThreadsDim[1]+1,
                            1, 0, stream1, NULL,
                            reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for (MaxBlockDimY + 1)");
    testStatus = false;
  }
  // Block dimension Z = Max Allowed + 1
  err = hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                            1,
                            1,
                            deviceProp.maxThreadsDim[2]+1, 0, stream1, NULL,
                            reinterpret_cast<void**>(&config1));
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed for (MaxBlockDimZ + 1)");
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

bool Module_GridBlock_Corner_Tests() {
  bool testStatus = true;
  HIPCHECK(hipSetDevice(0));
  hipError_t err;
  hipFunction_t DummyKernel;
  hipModule_t Module;
  hipStream_t stream1;
  hipDeviceptr_t *Ad;
  hipDevice_t device;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipCtxCreate(&context, 0, device));
#endif
  HIPCHECK(hipModuleLoad(&Module, fileName));
  HIPCHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIPCHECK(hipStreamCreate(&stream1));
  // Passing Max int value to block dimensions
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  unsigned int maxblockX = deviceProp.maxThreadsDim[0];
  unsigned int maxblockY = deviceProp.maxThreadsDim[1];
  unsigned int maxblockZ = deviceProp.maxThreadsDim[2];
#ifdef __HIP_PLATFORM_NVCC__
  unsigned int maxgridX = deviceProp.maxGridSize[0];
  unsigned int maxgridY = deviceProp.maxGridSize[1];
  unsigned int maxgridZ = deviceProp.maxGridSize[2];
#else
  unsigned int maxgridX = UINT32_MAX;
  unsigned int maxgridY = UINT32_MAX;
  unsigned int maxgridZ = UINT32_MAX;
#endif
  struct gridblockDim test[6] = {{1, 1, 1, maxblockX, 1, 1},
                                 {1, 1, 1, 1, maxblockY, 1},
                                 {1, 1, 1, 1, 1, maxblockZ},
                                 {maxgridX, 1, 1, 1, 1, 1},
                                 {1, maxgridY, 1, 1, 1, 1},
                                 {1, 1, maxgridZ, 1, 1, 1}};
  for (int i = 0; i < 6; i++) {
    err = hipModuleLaunchKernel(DummyKernel,
                                test[i].gridX,
                                test[i].gridY,
                                test[i].gridZ,
                                test[i].blockX,
                                test[i].blockY,
                                test[i].blockZ,
                                0,
                                stream1, NULL, NULL);
    if (err != hipSuccess) {
      printf("hipModuleLaunchKernel failed (%u, %u, %u) and (%u, %u, %u)",
      test[i].gridX, test[i].gridY, test[i].gridZ,
      test[i].blockX, test[i].blockY, test[i].blockZ);
      testStatus = false;
    }
  }
  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  hipCtxDestroy(context);
#endif
  return testStatus;
}

bool Module_WorkGroup_Test() {
  bool testStatus = true;
  HIPCHECK(hipSetDevice(0));
  hipError_t err;
  hipFunction_t DummyKernel;
  hipModule_t Module;
  hipStream_t stream1;
  hipDeviceptr_t *Ad;
  hipDevice_t device;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipCtxCreate(&context, 0, device));
#endif
  HIPCHECK(hipModuleLoad(&Module, fileName));
  HIPCHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIPCHECK(hipStreamCreate(&stream1));
  // Passing Max int value to block dimensions
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  double cuberootVal =
              cbrt(static_cast<double>(deviceProp.maxThreadsPerBlock));
  uint32_t cuberoot_floor = floor(cuberootVal);
  uint32_t cuberoot_ceil = ceil(cuberootVal);
  // Scenario: (block.x * block.y * block.z) <= Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  err = hipModuleLaunchKernel(DummyKernel,
                            1, 1, 1,
                            cuberoot_floor, cuberoot_floor, cuberoot_floor,
                            0, stream1, NULL, NULL);
  if (err != hipSuccess) {
    printf("hipModuleLaunchKernel failed block dimensions (%u, %u, %u)",
           cuberoot_floor, cuberoot_floor, cuberoot_floor);
    testStatus = false;
  }
  // Scenario: (block.x * block.y * block.z) > Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  err = hipModuleLaunchKernel(DummyKernel,
                            1, 1, 1,
                            cuberoot_ceil, cuberoot_ceil, cuberoot_ceil + 1,
                            0, stream1, NULL, NULL);
  if (err == hipSuccess) {
    printf("hipModuleLaunchKernel failed block dimensions (%u, %u, %u)",
           cuberoot_ceil, cuberoot_ceil, cuberoot_ceil);
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
  HipTest::parseStandardArguments(argc, argv, true);
  if (p_tests == 0x1) {
    testStatus = Module_Negative_tests();
  } else if (p_tests == 0x2) {
    testStatus = Module_GridBlock_Corner_Tests();
  } else if (p_tests == 0x3) {
    testStatus = Module_WorkGroup_Test();
  } else {
    printf("Invalid Test Case \n");
    exit(1);
  }
  if (testStatus) {
    passed();
  } else {
    failed("Test Failed!");
  }
}
