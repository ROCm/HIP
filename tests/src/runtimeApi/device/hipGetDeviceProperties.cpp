/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "test_common.h"

/**
 * Validates negative scenarios for hipGetDeviceProperties
 * scenario1: props = nullptr
 * scenario2: device = -1 (Invalid Device)
 * scenario3: device = Non Existing Device
 */
bool testInvalidParameters() {
  bool TestPassed = true;
  hipError_t ret;
  // props = nullptr
#ifndef __HIP_PLATFORM_NVCC__
  int device;
  HIPCHECK(hipGetDevice(&device));
  // this test case results in segmentation fault on NVCC
  ret = hipGetDeviceProperties(nullptr, device);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {props = nullptr} Failed \n");
  }
#endif
  hipDeviceProp_t prop;
  ret = hipGetDeviceProperties(&prop, -1);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {device = -1} Failed \n");
  }
  // device = Non Existing Device
  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  ret = hipGetDeviceProperties(&prop, deviceCount);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {device = Non Existing Device} Failed \n");
  }
  return TestPassed;
}

int main(int argc, char** argv) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = testInvalidParameters();
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
}
