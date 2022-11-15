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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/*
Testcase Scenarios :
Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Positive_RangeValidation - Test correct
execution of hipModuleOccupancyMaxPotentialBlockSizeWithFlags for diffrent parameter values
Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Negative_Parameters - Test unsuccessful
execution of hipModuleOccupancyMaxPotentialBlockSizeWithFlags api when parameters are invalid
*/
#include "occupancy_common.hh"

TEST_CASE("Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Negative_Parameters") {
  hipModule_t module;
  hipFunction_t function;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  // Common negative tests
  MaxPotentialBlockSizeNegative([&function](int* gridSize, int* blockSize) {
    return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, function, 0, 0,
                                                            hipOccupancyDefault);
  });

#ifdef __HIP_PLATFORM_NVIDIA__
  SECTION("Flag is invalid") {
    // Only default flag is supported
    HIP_CHECK_ERROR(
        hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize, &blockSize, function, 0, 0, 2),
        hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipModuleUnload(module));
}

TEST_CASE("Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Positive_RangeValidation") {
  hipDeviceProp_t devProp;
  hipModule_t module;
  hipFunction_t function;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [&function](int* gridSize, int* blockSize) {
          return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, function, 0,
                                                                  0, hipOccupancyDefault);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [&function, devProp](int* gridSize, int* blockSize) {
          return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
              gridSize, blockSize, function, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock,
              hipOccupancyDefault);
        },
        devProp.maxThreadsPerBlock);
  }

  HIP_CHECK(hipModuleUnload(module));
}
