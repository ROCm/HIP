/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :

 (TestCase 1)::
 1) Check api behavior by passing numBlocks ptr as NULL.
 2) Pass invalid kernel function/NULL and check the api behavior.
 3) Pass blockSize as zero and check appropriate error-code is returned.
 4) Pass shm as max size_t and validate error-code returned.
 5) Test occupancy api with other possible flags.

 (TestCase 2)::
 6) Validate range by making sure (numBlock * blockSize) doesn't exceed
 devProp.maxThreadsPerMultiProcessor.
 7) Check range of out param after passing valid dynSharedMemPerBlk.

 (TestCase 3)::
 8) Test case for using kernel function pointer with template.

*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST: %t --tests 1
 * TEST: %t --tests 2
 * TEST: %t --tests 3
 * HIT_END
 */

#include <iostream>
#include <limits>
#include "test_common.h"

__global__ void f1(float *a) { *a = 1.0; }

template <typename T>
__global__ void f2(T *a) { *a = 1; }

/**
 * Defines
 */
#define OccupancyDisableCachingOverride 0x01

/**
 * Performs argument validation
 */
bool argValidation() {
  bool TestPassed = true;
  hipError_t ret;
  int numBlock = 0, blockSize = 0;
  int gridSize = 0, defBlkSize = 32;

  // Get potential blocksize
  HIPCHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  // Validate each argument
  if ((ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(NULL, f1,
                   blockSize, 0)) != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropriate error value returned for"
        " numBlock(NULL). Error: '%s'(%d)\n", hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, NULL,
                   blockSize, 0);
  if (ret != hipErrorInvalidValue && ret != hipErrorInvalidDeviceFunction) {
    printf("ArgValidation : Inappropriate error value returned for"
        "  kernelfunc(NULL). numBlk %d, Error: '%s'(%d)\n", numBlock,
        hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  if ((ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock,
                  f1, 0, 0)) != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropriate error value returned for"
        " blksize(0), shm(0). numBlk %d, Error: '%s'(%d)\n", numBlock,
        hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  if ((ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock,
                  f1, 0, std::numeric_limits<std::size_t>::max()))
                  != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropriate error value returned for"
        " blksize(0), shm(max). numBlk %d, Error: '%s'(%d)\n", numBlock,
        hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  if ((ret = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlock,
                  f1, defBlkSize, 0, OccupancyDisableCachingOverride))
                  != hipSuccess) {
    printf("ArgValidation : Occupancy api with flags returned '%s'(%d)."
        " Expected to return hipSuccess(0)\n", hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  return TestPassed;
}


/**
 * Performs range validation on api output
 */
bool rangeValidation() {
  hipDeviceProp_t devProp;
  bool TestPassed = true;
  int numBlock = 0, blockSize = 0;
  int gridSize = 0;

  // Get potential blocksize
  HIPCHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  HIPCHECK(hipGetDeviceProperties(&devProp, 0));

  HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, f1,
              blockSize, 0));

  // Check if numBlocks and blockSize are within limits
  if ((numBlock <= 0) ||
     ((numBlock * blockSize) > devProp.maxThreadsPerMultiProcessor)) {
    printf("RangeValidation : numBlock %d returned not in range."
           "numblk(%d),blocksize(%d) and maxThrdsMP %d", numBlock, numBlock,
           blockSize, devProp.maxThreadsPerMultiProcessor);
    TestPassed &= false;
  }

  // Validate numBlock after passing dynSharedMemPerBlk
  HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, f1,
              blockSize, devProp.sharedMemPerBlock));

  // Check if numBlocks and blockSize are within limits
  if ((numBlock <= 0) ||
     ((numBlock * blockSize) > devProp.maxThreadsPerMultiProcessor)) {
    printf("RangeValidation : numBlock %d returned not in range."
          "numblk(%d),blocksize(%d),shm and maxThrdsMP %d", numBlock, numBlock,
           blockSize, devProp.maxThreadsPerMultiProcessor);
    TestPassed &= false;
  }

  return TestPassed;
}

/**
 * Test case for using kernel function pointer with template
 */
bool templateInvocation() {
  bool TestPassed = true;
  int blockSize = 32;
  int numBlock = 0;

  HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor<void(*)(int *)>
  (&numBlock, f2, blockSize, 0));
  if (!numBlock) {
    printf("TemplateInvocation : numBlock received as zero");
    TestPassed &= false;
  }

  return TestPassed;
}


int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;

  if (p_tests == 1) {
    TestPassed = argValidation();
  } else if (p_tests == 2) {
    TestPassed = rangeValidation();
  } else if (p_tests == 3) {
    TestPassed = templateInvocation();
  } else {
    printf("Didnt receive any valid option. Try options 1 to 3\n");
    TestPassed = false;
  }

  if (TestPassed) {
    passed();
  } else {
    failed("hipOccupancyMaxActiveBlocksPerMultiprocessor validation Failed!");
  }
}

