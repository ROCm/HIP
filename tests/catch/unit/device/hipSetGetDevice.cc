/*
 * Copyright (c) 2021-present Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Verifies functionality of hipSetDevice/hipGetDevice api.
 * -- Basic Test to set and get valid device numbers.
 */

#include <hip_test_common.hh>

TEST_CASE("Unit_hipSetDevice_BasicSetGet") {
  int numDevices = 0;
  int device;
  int validateCount = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices != 0);

  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipGetDevice(&device));
    if (device == i) {
      validateCount+= 1;
    }
  }

  REQUIRE(numDevices == validateCount);
}
