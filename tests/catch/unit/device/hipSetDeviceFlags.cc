/*
Copyright (c) 2021-Present Advanced Micro Devices, Inc. All rights reserved.
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
/*
 * Conformance test for checking functionality of
 * hipError_t hipGetDeviceFlags(unsigned int* flags);
 * hipError_t hipSetDeviceFlags(unsigned flags);
 */
/**
 * hipGetDeviceFlags and hipSetDeviceFlags tests.
 * Scenario1: Validates if hipGetDeviceFlags returns hip error code for
 *  flags = nullptr.
 * Scenario2: Validates if hipSetDeviceFlags returns hip error code for
 *  invalid flag.
 * Scenario3: Validates if flags = hipDeviceScheduleSpin|hipDeviceScheduleYield
 *  |hipDeviceScheduleBlockingSync|hipDeviceScheduleAuto|hipDeviceMapHost
 *  |hipDeviceLmemResizeToMax returned by hipGetDeviceFlags.
 */
TEST_CASE("Unit_hipGetDeviceFlags_NegTst") {
  // Scenario1
  SECTION("flags is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipGetDeviceFlags(nullptr));
  }

  // Scenario2
  SECTION("flags value is invalid") {
    REQUIRE_FALSE(hipSuccess == hipSetDeviceFlags(0xffff));
  }
}


TEST_CASE("Unit_hipGetDeviceFlags_FuncTst") {
  unsigned flag = 0;

  // Scenario3
  SECTION("Check flag value") {
    HIP_CHECK(hipGetDeviceFlags(&flag));
    bool checkFlg = false;
    checkFlg = ((flag != hipDeviceScheduleSpin) &&
                (flag != hipDeviceScheduleYield) &&
                (flag != hipDeviceScheduleBlockingSync) &&
                (flag != hipDeviceScheduleAuto) &&
                (flag != hipDeviceMapHost) &&
                (flag != hipDeviceLmemResizeToMax));
    REQUIRE_FALSE(checkFlg);
  }

  SECTION("Set flag value") {
    auto devNo = GENERATE(range(0, HipTest::getDeviceCount()));
    flag = 0;
    HIP_CHECK(hipSetDevice(devNo));
    auto bitmap = GENERATE(range(0, 4));
    flag = 1 << bitmap;
    HIP_CHECK(hipSetDeviceFlags(flag));
  }
}
