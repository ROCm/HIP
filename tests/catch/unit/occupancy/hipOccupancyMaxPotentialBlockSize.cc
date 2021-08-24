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
#include <hip_test_common.hh>

static __global__ void f1(float *a) { *a = 1.0; }

template <typename T>
static __global__ void f2(T *a) { *a = 1; }

TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_Negative") {
  hipError_t ret;
  int blockSize = 0;
  int gridSize = 0;

  // Validate each argument
  ret = hipOccupancyMaxPotentialBlockSize(NULL, &blockSize, f1, 0, 0);
  REQUIRE(ret != hipSuccess);

  ret = hipOccupancyMaxPotentialBlockSize(&gridSize, NULL, f1, 0, 0);
  REQUIRE(ret != hipSuccess);

#ifndef __HIP_PLATFORM_NVIDIA__
  // nvcc doesnt support kernelfunc(NULL) for api
  ret = hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, NULL, 0, 0);
  REQUIRE(ret != hipSuccess);
#endif
}

TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_rangeValidation") {
  hipDeviceProp_t devProp;
  int blockSize = 0;
  int gridSize = 0;

  // Get potential blocksize
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  // Check if blockSize doen't exceed maxThreadsPerBlock
  REQUIRE(gridSize > 0); REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= devProp.maxThreadsPerBlock);

  // Pass dynSharedMemPerBlk, blockSizeLimit and check out param
  blockSize = 0;
  gridSize = 0;

  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1,
           devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock));

  // Check if blockSize doen't exceed maxThreadsPerBlock
  REQUIRE(gridSize > 0); REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= devProp.maxThreadsPerBlock);

}

TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_templateInvocation") {
  int gridSize = 0, blockSize = 0;

  HIP_CHECK(hipOccupancyMaxPotentialBlockSize<void(*)(int *)>(&gridSize,
                       &blockSize, f2, 0, 0));
  REQUIRE(gridSize > 0);
  REQUIRE(blockSize > 0);
}

