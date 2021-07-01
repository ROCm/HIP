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
 * hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
 */
#include <hip_test_common.hh>

/**
 * hipDeviceComputeCapability tests
 * Scenario1: Validates if &major = nullptr returns error code
 * Scenario2: Validates if &minor = nullptr returns error code
 * Scenario3: Check if Major and Minor Versions are valid
 */

// Scenario 1 and 2
TEST_CASE("Unit_hipDeviceComputeCapability_NegTst") {
  int major, minor, numDevices;
  hipDevice_t device;

  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    HIP_CHECK(hipDeviceGet(&device, 0));

    // Scenario1
    SECTION("major is nullptr") {
      REQUIRE_FALSE(hipDeviceComputeCapability(nullptr, &minor, device)
                          == hipSuccess);
    }

    // Scenario2
    SECTION("minor is nullptr") {
      REQUIRE_FALSE(hipDeviceComputeCapability(&major, nullptr, device)
                          == hipSuccess);
    }
  } else {
    WARN("Test skipped as no gpu devices available");
  }
}

// Scenario 3 : Check whether major and minor version value is valid.
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
