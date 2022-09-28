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
Testcase Scenarios :
Unit_hipDeviceGetUuid_Positive - Check if hipDeviceGetUuid api returns valid UUID
Unit_hipDeviceGetUuid_Negative - Test unsuccessful execution of hipDeviceGetUuid when nullptr
                                 or invalid device is set as input parameter
*/
/*
 * Conformance test for checking functionality of
 * hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device);
 */
#include <hip_test_common.hh>
#include <cstring>
#include <cstdio>

/**
 * hipDeviceGetUuid positive test
 * Scenario1: Validates the returned UUID
 */
TEST_CASE("Unit_hipDeviceGetUuid_Positive") {
  int numDevices = 0;
  hipDevice_t device;
  hipUUID uuid;

  const int deviceId = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipDeviceGet(&device, deviceId));

  // Scenario 1
  HIP_CHECK(hipDeviceGetUuid(&uuid, device));
  REQUIRE(strcmp(uuid.bytes, "") != 0);
}

/**
 * hipDeviceGetUuid negative tests
 * Scenario2: Validates returned error code for UUID = nullptr
 * Scenario3: Validates returned error code for invalid device
 */
TEST_CASE("Unit_hipDeviceGetUuid_Negative") {
  int numDevices = 0;
  hipUUID uuid;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  std::vector<hipDevice_t> devices(numDevices);
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&devices[i], i));
  }

  if (numDevices > 0) {
    // Scenario 2
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(nullptr, devices[0]));
    // Scenario 3
    hipDevice_t badDevice = devices.back() + 1;

    constexpr size_t timeout = 100;
    size_t timeoutCount = 0;
    while (std::find(std::begin(devices), std::end(devices), badDevice) != std::end(devices)) {
      badDevice += 1;
      timeoutCount += 1;
      REQUIRE(timeoutCount < timeout);  // give up after a while
    }

    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(&uuid, badDevice));
  }
}
