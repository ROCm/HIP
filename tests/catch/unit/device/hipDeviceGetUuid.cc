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
 * hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device);
 */
#include <hip_test_common.hh>
#include <cstring>
#include <cstdio>

/**
 * hipDeviceGetUuid tests
 * Scenario1: Validates the returned UUID
 * Scenario2: Validates returned error code for UUID = nullptr
 * Scenario3 & 4: Validates returned error code for invalid device
 */
TEST_CASE("Unit_hipDeviceGetUuid") {
  int numDevices = 0;
  hipDevice_t device;
  hipUUID uuid;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    // Scenario 1
    HIP_CHECK(hipDeviceGetUuid(&uuid, device));
    REQUIRE_FALSE(!strcmp(uuid.bytes, ""));
    // Scenario 2
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(nullptr, device));
  }
  // Scenario 3
  REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(&uuid, -1));
  // Scenario 4
  REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(&uuid, numDevices));
}
