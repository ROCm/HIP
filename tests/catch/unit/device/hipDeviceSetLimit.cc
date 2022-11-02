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
 * Conformance test for checking functionality of
 * hipError_t hipDeviceSetLimit ( enum hipLimit_t limit, size_t value );
 */
#include <hip_test_common.hh>

// Currently the HIGHER_VALUE is fixed to 16 KB based on currently
// set maximum value for hipLimitStackSize. In future, this value
// might need to change to avoid test case failure. In the same way
// LOWER_VALUE is empirically determined.
#define HIGHER_VALUE (16*1024)  // 16 KB
#define LOWER_VALUE (512)  // 512 bytes

/**
 * hipDeviceSetLimit tests =>
 */
static bool testSetLimitFunc(hipLimit_t limit_to_test) {
  size_t value = 0;
  size_t setValue = LOWER_VALUE;
  // Set Value to low Value.
  HIP_CHECK(hipDeviceSetLimit(limit_to_test, setValue));
  HIP_CHECK(hipDeviceGetLimit(&value, limit_to_test));
  // The returned value could be rounded to maximum or minimum
  REQUIRE(value >= LOWER_VALUE);
  // Set Value to High value.
  setValue = HIGHER_VALUE;
  HIP_CHECK(hipDeviceSetLimit(limit_to_test, setValue));
  HIP_CHECK(hipDeviceGetLimit(&value, limit_to_test));
  // The returned value could be rounded to maximum or minimum
  REQUIRE(value >= HIGHER_VALUE);
  return true;
}

/**
 * hipDeviceSetLimit tests =>
 *
 * Scenario1: Single device Set-Get test for hipLimitStackSize flag.
 *
 * Scenario2: Single device Set-Get test for hipLimitPrintfFifoSize flag.
 *
 * Scenario3: Single device Set-Get test for hipLimitMallocHeapSize flag.
 *
 * Scenario4: Multidevice Set-Get test for all the flags
 *
 * Scenario5: Negative Scenario - Invalid flag value
 */
TEST_CASE("Unit_hipDeviceSetLimit_SetGet") {
  size_t value = 0;
  // Scenario1
  SECTION("Set Get Test hipLimitStackSize") {
    REQUIRE(true == testSetLimitFunc(hipLimitStackSize));
  }
#if HT_NVIDIA
  // Scenario2
  SECTION("Set Get Test hipLimitPrintfFifoSize") {
    REQUIRE(true == testSetLimitFunc(hipLimitPrintfFifoSize));
  }
  // Scenario3
  SECTION("Set Get Test hipLimitMallocHeapSize") {
    REQUIRE(true == testSetLimitFunc(hipLimitMallocHeapSize));
  }
#endif
  // Scenario4
  SECTION("Multi device Set-Get test for all flags") {
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    for (int dev = 0; dev < numDevices; dev++) {
      HIP_CHECK(hipSetDevice(dev));
      REQUIRE(true == testSetLimitFunc(hipLimitStackSize));
#if HT_NVIDIA
      REQUIRE(true == testSetLimitFunc(hipLimitPrintfFifoSize));
      REQUIRE(true == testSetLimitFunc(hipLimitMallocHeapSize));
#endif
    }
  }
  // Scenario5
  SECTION("Negative Scenario: Invalid Flag") {
    HIP_CHECK(hipDeviceGetLimit(&value, hipLimitMallocHeapSize));
    REQUIRE(hipErrorInvalidValue == hipDeviceSetLimit(
    static_cast<hipLimit_t>(0xffff), value/2));
  }
}
