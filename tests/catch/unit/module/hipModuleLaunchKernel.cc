/*
   Copyright (c) 2022 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
  2. hipModuleLaunchKernel Work Group tests =>
     - (block.x * block.y * block.z) <= Work Group Size
       where block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
     - (block.x * block.y * block.z) > Work Group Size
       where block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
 */

#include <math.h>
#include <hip_test_common.hh>

#define fileName "module_kernels.code"
#define matmulK "matmulK"
#define SixteenSec "SixteenSecKernel"
#define KernelandExtra "KernelandExtraParams"
#define FourSec "FourSecKernel"
#define TwoSec "TwoSecKernel"
#define dummyKernel "EmptyKernel"

struct gridblockDim {
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int blockZ;
};

/*
This testcase verifies the negative scenarios of
hipModuleLaunchKernel API
*/
TEST_CASE("Unit_hipModuleLaunchKernel_Negative") {
  HIP_CHECK(hipSetDevice(0));
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
  hipStream_t stream1;
  hipDeviceptr_t *Ad{nullptr};
  CTX_CREATE()

  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&MultKernel, Module, matmulK));
  HIP_CHECK(hipModuleGetFunction(&KernelandExtraParamKernel,
                                Module, KernelandExtra));
  void *config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  void *params[] = {Ad};
  HIP_CHECK(hipStreamCreate(&stream1));
  SECTION("Passing nullptr to kernel function") {
    REQUIRE(hipModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0,
                                  stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Passing Max int value to block dim") {
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                                  std::numeric_limits<uint32_t>::max(),
                                  std::numeric_limits<uint32_t>::max(),
                                  std::numeric_limits<uint32_t>::max(),
                                  0, stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }


  SECTION("Passing 0 to all value dim") {
    REQUIRE(hipModuleLaunchKernel(MultKernel, 0, 0, 0,
                                  0,
                                  0,
                                  0, 0,
                                  stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Passing 0 for X dim") {
    REQUIRE(hipModuleLaunchKernel(MultKernel, 0, 1, 1,
                                  0,
                                  1,
                                  1, 0,
                                  stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }


  SECTION("Passing 0 for Y dim") {
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 0, 1,
                                  1,
                                  0,
                                  1, 0,
                                  stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Passing 0 for Z dim") {
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 1, 0,
                                  1,
                                  1,
                                  0, 0,
                                  stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Passing both kernel and extra params") {
    REQUIRE(hipModuleLaunchKernel(KernelandExtraParamKernel, 1, 1, 1, 1,
                                  1, 1, 0, stream1,
                                  reinterpret_cast<void**>(&params),
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Passing more than maxthreadsperblock to block dim") {
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 0);
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                                  deviceProp.maxThreadsPerBlock+1,
                                  deviceProp.maxThreadsPerBlock+1,
                                  deviceProp.maxThreadsPerBlock+1, 0,
                                  stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Block dim X is more than max allowed") {
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 0);
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                                  deviceProp.maxThreadsDim[0]+1,
                                  1,
                                  1, 0, stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Block dim Y is more than max allowed") {
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 0);
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                                  1,
                                  deviceProp.maxThreadsDim[1]+1,
                                  1, 0, stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Block dim Z is more than max allowed") {
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 0);
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                                  1,
                                  1,
                                  deviceProp.maxThreadsDim[2]+1,
                                  0, stream1, NULL,
                                  reinterpret_cast<void**>(&config1))
                                  != hipSuccess);
  }

  SECTION("Block invalid config to extra params") {
    void *config3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                       HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                       HIP_LAUNCH_PARAM_END};
    REQUIRE(hipModuleLaunchKernel(MultKernel, 1, 1, 1,
                                  1, 1, 1, 0, stream1,
                                  NULL,
                                  reinterpret_cast<void**>(&config3))
                                  != hipSuccess);
  }

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}

/*
This testcase verifies the work group scenarios of
hipModuleLaunchKernel API
*/
TEST_CASE("Unit_hipModuleLaunchKernel_WorkGroup") {
  HIP_CHECK(hipSetDevice(0));
  hipFunction_t DummyKernel;
  hipModule_t Module;
  hipStream_t stream1;
  CTX_CREATE()

  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIP_CHECK(hipStreamCreate(&stream1));
  // Passing Max int value to block dimensions
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  double cuberootVal =
              cbrt(static_cast<double>(deviceProp.maxThreadsPerBlock));
  uint32_t cuberoot_floor = floor(cuberootVal);
  uint32_t cuberoot_ceil = ceil(cuberootVal);
  // Scenario: (block.x * block.y * block.z) <= Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  HIP_CHECK(hipModuleLaunchKernel(DummyKernel,
                            1, 1, 1,
                            cuberoot_floor, cuberoot_floor, cuberoot_floor,
                            0, stream1, NULL, NULL));
  // Scenario: (block.x * block.y * block.z) > Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  REQUIRE(hipModuleLaunchKernel(DummyKernel,
                            1, 1, 1,
                            cuberoot_ceil, cuberoot_ceil, cuberoot_ceil + 1,
                            0, stream1, NULL, NULL) != hipSuccess);
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}
