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

/**
 * Defines
 */
#define OccupancyDisableCachingOverride 0x01

TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Negative") {
  hipError_t ret;
  int numBlock = 0, blockSize = 0;
  int gridSize = 0, defBlkSize = 32;

  // Get potential blocksize
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  // Validate each argument
  ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(NULL, f1, blockSize, 0);
  REQUIRE(ret != hipSuccess);

  ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, NULL, blockSize, 0);
  REQUIRE(ret != hipSuccess);

  ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, f1, 0, 0);
  REQUIRE(ret != hipSuccess);

  ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, f1, 0,
                                         std::numeric_limits<std::size_t>::max());
  REQUIRE(ret != hipSuccess);

  ret = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlock, f1,
                                  defBlkSize, 0, OccupancyDisableCachingOverride);
  REQUIRE(ret == hipSuccess);
}

TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_rangeValidation") {
  hipDeviceProp_t devProp;
  int numBlock = 0, blockSize = 0;
  int gridSize = 0;

  // Get potential blocksize
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, f1, blockSize, 0));

  // Check if numBlocks and blockSize are within limits
  REQUIRE(numBlock > 0);
  REQUIRE((numBlock * blockSize) <= devProp.maxThreadsPerMultiProcessor);

  // Validate numBlock after passing dynSharedMemPerBlk
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, f1, blockSize,
                                                           devProp.sharedMemPerBlock));

  // Check if numBlocks and blockSize are within limits
  REQUIRE(numBlock > 0);
  REQUIRE((numBlock * blockSize) <= devProp.maxThreadsPerMultiProcessor);
}

TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_templateInvocation") {
  int blockSize = 32;
  int numBlock = 0;

  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor<void(*)(int *)>
                                                  (&numBlock, f2, blockSize, 0));
  REQUIRE(numBlock > 0);
}

