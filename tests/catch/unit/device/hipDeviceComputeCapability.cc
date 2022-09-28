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
Unit_hipDeviceComputeCapability_ValidateVersion - Check if hipDeviceComputeCapability api returns valid Major and Minor versions
Unit_hipDeviceComputeCapability_Negative - Test unsuccessful execution of hipDeviceComputeCapability when nullptr
                                           or invalid device is set as input parameter
*/

/*
 * Conformance test for checking functionality of
 * hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
 */
#include <hip_test_common.hh>

/**
 * hipDeviceComputeCapability negative tests
 * Scenario1: Validates if &major = nullptr returns error code
 * Scenario2: Validates if &minor = nullptr returns error code
 * Scenario3: Validates if device is invalid device returns error code
 */
TEST_CASE("Unit_hipDeviceComputeCapability_Negative") {
  int major, minor, numDevices;

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  std::vector<hipDevice_t> devices(numDevices);
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&devices[i], i));
  }

  if (numDevices > 0) {

    // Scenario1
    SECTION("major is nullptr") {
      REQUIRE_FALSE(hipDeviceComputeCapability(nullptr, &minor, devices[0])
                          == hipSuccess);
    }

    // Scenario2
    SECTION("minor is nullptr") {
      REQUIRE_FALSE(hipDeviceComputeCapability(&major, nullptr, devices[0])
                          == hipSuccess);
    }

    // Scenario3
    SECTION("Invalid Device") {
      hipDevice_t badDevice = devices.back() + 1;

      constexpr size_t timeout = 100;
      size_t timeoutCount = 0;
      while (std::find(std::begin(devices), std::end(devices), badDevice) != std::end(devices)) {
        badDevice += 1;
        timeoutCount += 1;
        REQUIRE(timeoutCount < timeout);  // give up after a while
      }

      REQUIRE_FALSE(hipDeviceComputeCapability(&major, &minor, badDevice)
                          == hipSuccess);
    }
  } else {
    WARN("Test skipped as no gpu devices available");
  }
}

// Scenario 4 : Check whether major and minor version value is valid.
TEST_CASE("Unit_hipDeviceComputeCapability_ValidateVersion") {
  int major, minor;
  hipDevice_t device;
  int numDevices = -1;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));
    REQUIRE(major >= 0);
    REQUIRE(minor >= 0);
  }
}
