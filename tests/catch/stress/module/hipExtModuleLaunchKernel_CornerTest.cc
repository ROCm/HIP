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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */

/*
Test Scenario
hipExtModuleLaunchKernel API verifying Corner Scenarios for Grid and Block dimensions
*/

#include "hip_test_common.hh"
#include "hip_test_kernels.hh"
#include "hip/hip_ext.h"

#define fileName "kernels.code"
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
This testcase verifies hipExtModuleLaunchKernel API Corner
cases
*/
TEST_CASE("Stress_hipExtModuleLaunchKernel_CornerCases") {
  hipModule_t Module;
  hipFunction_t DummyKernel;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  constexpr auto gridblocksize{6};
  struct {
  } args;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  size_t size = sizeof(args);
  void *config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                     HIP_LAUNCH_PARAM_END};
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  unsigned int maxblockX = deviceProp.maxThreadsDim[0];
  unsigned int maxblockY = deviceProp.maxThreadsDim[1];
  unsigned int maxblockZ = deviceProp.maxThreadsDim[2];
  struct gridblockDim test[gridblocksize] = {{1, 1, 1, maxblockX, 1, 1},
                                            {1, 1, 1, 1, maxblockY, 1},
                                            {1, 1, 1, 1, 1, maxblockZ},
                                            {UINT32_MAX, 1, 1, 1, 1, 1},
                                            {1, UINT32_MAX, 1, 1, 1, 1},
                                            {1, 1, UINT32_MAX, 1, 1, 1}};

  // Launching kernel with corner cases in grid and block dimensions
  for (int i = 0; i < gridblocksize; i++) {
    HIP_CHECK(hipExtModuleLaunchKernel(DummyKernel,
                                       test[i].gridX,
                                       test[i].gridY,
                                       test[i].gridZ,
                                       test[i].blockX,
                                       test[i].blockY,
                                       test[i].blockZ,
                                       0,
                                       stream, NULL,
                                       reinterpret_cast<void**>(&config1),
                                       nullptr, nullptr, 0));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
