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
 1) Pass gridSize as NULL and check appropriate error-code is returned.
 2) Pass blockSize as NULL and check appropriate error-code is returned.
 3) Pass invalid kernel function/NULL and check the api behavior.

 (TestCase 2)::
 4) Validate range by making sure blockSize returned by api doesn't exceed
 devProp.maxThreadsPerBlock.
 5) Pass dynSharedMemPerBlk, blockSizeLimit and check out parameter range.

 (TestCase 3)::
 6) Test case for using kernel function pointer with template.

*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
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
 * Performs argument validation
 */
bool argValidation() {
  bool TestPassed = true;
  hipError_t ret;
  int blockSize = 0;
  int gridSize = 0;

  // Validate each argument
  if ((ret = hipOccupancyMaxPotentialBlockSize(NULL, &blockSize,
             f1, 0, 0)) != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropritate error value returned for"
           " gridSize(NULL). blksize rcvd %d, Error: '%s'(%d)\n",
           blockSize, hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  if ((ret = hipOccupancyMaxPotentialBlockSize(&gridSize, NULL,
                  f1, 0, 0)) != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropritate error value returned for"
           "  blockSize(NULL). gridSize rcvd %d, Error: '%s'(%d)\n",
           gridSize, hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

#ifndef __HIP_PLATFORM_NVCC__
  // nvcc doesnt support kernelfunc(NULL) for api
  ret = hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, NULL, 0, 0);
  if (ret != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropritate error value returned for"
           " kernelfunc(NULL). gridSize %d, blkSize %d, Error: '%s'(%d)\n",
           gridSize, blockSize, hipGetErrorString(ret), ret);
    TestPassed &= false;
  }
#endif

  return TestPassed;
}


/**
 * Performs range validation on api output
 */
bool rangeValidation() {
  hipDeviceProp_t devProp;
  bool TestPassed = true;
  int blockSize = 0;
  int gridSize = 0;

  // Get potential blocksize
  HIPCHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  HIPCHECK(hipGetDeviceProperties(&devProp, 0));

  // Check if blockSize doen't exceed maxThreadsPerBlock
  if ((gridSize <= 0) || (blockSize <= 0) ||
     (blockSize > devProp.maxThreadsPerBlock)) {
    printf("RangeValidation : grdSize %d/blkSize %d returned not in range(%d)",
                              gridSize, blockSize, devProp.maxThreadsPerBlock);
    TestPassed &= false;
  }

  // Pass dynSharedMemPerBlk, blockSizeLimit and check out param
  blockSize = 0;
  gridSize = 0;

  HIPCHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1,
           devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock));

  if ((gridSize <= 0) || (blockSize <= 0) ||
     (blockSize > devProp.maxThreadsPerBlock)) {
    printf("RangeValidation(Shm,TPB) : grdSize %d/blkSize %d returned"
          "not in range(%d)", gridSize, blockSize, devProp.maxThreadsPerBlock);
    TestPassed &= false;
  }


  return TestPassed;
}

/**
 * Test case for using kernel function pointer with template
 */
bool templateInvocation() {
  bool TestPassed = true;
  int gridSize = 0, blockSize = 0;
  int numBlock = 0;

  HIPCHECK(hipOccupancyMaxPotentialBlockSize<void(*)(int *)>(&gridSize,
                       &blockSize, f2, 0, 0));
  if (!gridSize || !blockSize) {
    printf("TemplateInvocation : gridSize/blockSize received as zero");
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
    failed("hipOccupancyMaxPotentialBlockSize validation Failed!");
  }
}

