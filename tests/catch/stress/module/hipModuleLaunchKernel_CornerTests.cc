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
hipModuleLaunchKernel API verifying Corner Scenarios for Grid and Block dimensions
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
This testcase verifies hipModuleLaunchKernel API Corner
cases
*/
TEST_CASE("Stress_hipModuleLaunchKernel_CornerCases") {
  HIP_CHECK(hipSetDevice(0));
  hipStream_t stream1;
  CTX_CREATE()
  hipModule_t Module;
  hipFunction_t DummyKernel;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIP_CHECK(hipStreamCreate(&stream1));

  // Passing Max int value to block dimensions
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  unsigned int maxblockX = deviceProp.maxThreadsDim[0];
  unsigned int maxblockY = deviceProp.maxThreadsDim[1];
  unsigned int maxblockZ = deviceProp.maxThreadsDim[2];
#if HT_NVIDIA
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
    HIP_CHECK(hipModuleLaunchKernel(DummyKernel,
                                test[i].gridX,
                                test[i].gridY,
                                test[i].gridZ,
                                test[i].blockX,
                                test[i].blockY,
                                test[i].blockZ,
                                0,
                                stream1, NULL, NULL));
    }
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY();
}
