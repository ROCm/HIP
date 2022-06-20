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
 * Conformance test for checking functionality of
 * hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
 */
#include <hip_test_common.hh>


/**
 * hipDeviceTotalMem tests
 * Scenario1: Validates if bytes = nullptr returns hip error code.
 * Scenario2: Validates if error code is returned for device = -1.
 * Scenario3: Validates if error code is returned for device = deviceCount.
 * Scenario4: Compare total memory size with hipDeviceProp_t.totalGlobalMem for each device.
 */
TEST_CASE("Unit_hipDeviceTotalMem_NegTst") {
#if HT_NVIDIA
  HIP_CHECK(hipInit(0));
#endif
  // Scenario 1
  SECTION("bytes is nullptr") {
    HIP_CHECK_ERROR(hipDeviceTotalMem(nullptr, 0), hipErrorInvalidValue);
  }

  size_t totMem;
  // Scenario 2
  SECTION("device is -1") {
    HIP_CHECK_ERROR(hipDeviceTotalMem(&totMem, -1), hipErrorInvalidDevice);
  }

  // Scenario 3
  SECTION("pi is nullptr") {
    int numDevices;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    size_t totMem;
    HIP_CHECK_ERROR(hipDeviceTotalMem(&totMem, numDevices), hipErrorInvalidDevice);
  }
}

// Scenario 4
TEST_CASE("Unit_hipDeviceTotalMem_ValidateTotalMem") {
  size_t totMem;
  int numDevices = 0;

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices != 0);

  hipDevice_t device;
  hipDeviceProp_t prop;
  auto devNo = GENERATE_COPY(range(0, numDevices));
  totMem = 0;
  HIP_CHECK(hipDeviceGet(&device, devNo));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  HIP_CHECK(hipDeviceTotalMem(&totMem, device));

  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));

  REQUIRE(totMem == prop.totalGlobalMem);
  REQUIRE(total == totMem);
}

TEST_CASE("Unit_hipDeviceTotalMem_NonSelectedDevice") {
  auto deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Multi Device Test, will not run on single gpu systems. Skipping.");
  }

  for (int i = 1; i < deviceCount; i++) {
    HIP_CHECK(hipSetDevice(i - 1));
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, i));

    size_t totMem = 0;
    hipDeviceProp_t prop;
    HIP_CHECK(hipDeviceTotalMem(&totMem, device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    REQUIRE(totMem == prop.totalGlobalMem);
  }
}
