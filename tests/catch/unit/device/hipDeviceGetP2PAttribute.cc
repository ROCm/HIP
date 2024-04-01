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

#include <cstdlib>
#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include "hip/hip_runtime_api.h"
#include <hip_test_process.hh>
#include <string>

/**
 * @brief Test all possible combination of attributes and devices for hipDeviceGetP2PAttribute.
 *        Verify that the output is within the range of acceptable values.
 *
 */
TEST_CASE("Unit_hipDeviceGetP2PAttribute_Basic") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-119");
  return;
#else

  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  hipDeviceP2PAttr attribute =
      GENERATE(hipDevP2PAttrPerformanceRank, hipDevP2PAttrAccessSupported,
               hipDevP2PAttrNativeAtomicSupported, hipDevP2PAttrHipArrayAccessSupported);

  /* Test all combinations of devices in the system */
  for (int srcDevice = 0; srcDevice < deviceCount; ++srcDevice) {
    for (int dstDevice = 0; dstDevice < deviceCount; ++dstDevice) {
      if (srcDevice != dstDevice) {
        int value{-1};
        HIP_CHECK(hipDeviceGetP2PAttribute(&value, attribute, srcDevice, dstDevice));
        INFO("hipDeviceP2PAttr: " << attribute << "\nsrcDevice: " << srcDevice
                                  << "\ndstDevice: " << dstDevice << "\nValue: " << value);
        if (attribute == hipDevP2PAttrPerformanceRank) {
          REQUIRE(value >= 0);
        } else {
          REQUIRE((value == 0 || value == 1));
        }
      }
    }
  }
#endif
}

/**
 * @brief Negative test scenarios for hipDeviceGetP2PAttribute
 *
 */
TEST_CASE("Unit_hipDeviceGetP2PAttribute_Negative") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-122");
  return;
#else

  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int value;
  int validSrcDevice = 0;
  int validDstDevice = 1;
  hipDeviceP2PAttr validAttr = hipDevP2PAttrAccessSupported;

  SECTION("Nullptr value") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(nullptr, validAttr, validSrcDevice, validDstDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid attribute") {
    hipDeviceP2PAttr invalidAttr = static_cast<hipDeviceP2PAttr>(10);
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, invalidAttr, validSrcDevice, validDstDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Device is -1") {
    int invalidDevice = -1;
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, invalidDevice, validDstDevice),
                    hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, validSrcDevice, invalidDevice),
                    hipErrorInvalidDevice);
  }

  SECTION("Device is out of bounds") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    REQUIRE_FALSE(deviceCount == 0);

    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, deviceCount, validDstDevice),
                    hipErrorInvalidDevice);
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, validSrcDevice, deviceCount),
                    hipErrorInvalidDevice);
  }

  SECTION("Source and destination devices are the same") {
    HIP_CHECK_ERROR(hipDeviceGetP2PAttribute(&value, validAttr, validSrcDevice, validSrcDevice),
                    hipErrorInvalidDevice);
  }

  /* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars */
  SECTION("Hidden devices using environment variables") {
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("") == hipSuccess);
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("0") == hipErrorInvalidDevice);
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("1") == hipErrorInvalidDevice);
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("0,1") == hipSuccess);
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("-1,0") == hipErrorNoDevice);
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("0,-1") == hipErrorInvalidDevice);
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("0,1,-1") == hipSuccess);
    REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("0,-1,1") == hipErrorInvalidDevice);

    if (deviceCount > 2) {
      REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("2,1") == hipSuccess);
      REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("2") == hipErrorInvalidDevice);
    } else {
      REQUIRE(hip::SpawnProc("hipDeviceGetP2PAttribute").run("2,1") == hipErrorNoDevice);
    }
  }
#endif
}