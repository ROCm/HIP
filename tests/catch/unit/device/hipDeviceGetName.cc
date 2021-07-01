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
 * hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
 */
#include <hip_test_common.hh>
#include <cstring>
#include <cstdio>

#define LEN 256
/**
 * hipDeviceGetName tests
 * Scenario1: Validates the name string with hipDeviceProp_t.name[256]
 * Scenario2: Validates returned error code for name = nullptr
 * Scenario3: Validates returned error code for len = 0
 * Scenario4: Validates returned error code for len < 0
 */
TEST_CASE("Unit_hipDeviceGetName-NegTst") {
  int numDevices = 0;
  char name[LEN];
  hipDevice_t device;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(hipDeviceGetName(name, LEN, device));
    // Scenario2
    CHECK_FALSE(hipSuccess == hipDeviceGetName(nullptr, LEN, device));
#if HT_AMD
    // These test scenarios fail on NVIDIA.
    // Scenario3
    CHECK_FALSE(hipSuccess == hipDeviceGetName(name, 0, device));
    // Scenario4
    CHECK_FALSE(hipSuccess == hipDeviceGetName(name, -1, device));
#endif
  }
}

TEST_CASE("Unit_hipDeviceGetName-CheckPropName") {
  int numDevices = 0;
  char name[LEN];
  hipDevice_t device;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(hipDeviceGetName(name, LEN, device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    // Scenario1
    CHECK_FALSE(0 != strcmp(name, prop.name));
  }
}
