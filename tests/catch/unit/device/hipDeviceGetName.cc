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
 * hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
 */
#include <cstddef>
#include <hip_test_common.hh>
#include <cstring>
#include <cstdio>
#include <array>
#include <algorithm>
#include <iterator>

constexpr size_t LEN = 256;

/**
 * hipDeviceGetName tests
 * Scenario1: Validates the name string with hipDeviceProp_t.name[256]
 * Scenario2: Validates returned error code for name = nullptr
 * Scenario3: Validates returned error code for len = 0
 * Scenario4: Validates returned error code for len < 0
 * Scenario5: Validates returned error code for an invalid device
 * Scenario6: Validates partially filling the name into a char array
 */
TEST_CASE("Unit_hipDeviceGetName_NegTst") {
  std::array<char, LEN> name;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  std::vector<hipDevice_t> devices(numDevices);
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&devices[i], i));
  }

  SECTION("Valid Device") {
    const auto device = GENERATE_COPY(from_range(std::begin(devices), std::end(devices)));

    SECTION("Nullptr for name argument") {
      // Scenario2
      HIP_CHECK_ERROR(hipDeviceGetName(nullptr, name.size(), device), hipErrorInvalidValue);
    }
#if HT_AMD
    // These test scenarios fail on NVIDIA.
    SECTION("Zero name length") {
      // Scenario3
      HIP_CHECK_ERROR(hipDeviceGetName(name.data(), 0, device), hipErrorInvalidValue);
    }

    SECTION("Negative name length") {
      // Scenario4
      HIP_CHECK_ERROR(hipDeviceGetName(name.data(), -1, device), hipErrorInvalidValue);
    }
#endif
  }
  SECTION("Invalid Device") {
    hipDevice_t badDevice = devices.back() + 1;

    constexpr size_t timeout = 100;
    size_t timeoutCount = 0;
    while (std::find(std::begin(devices), std::end(devices), badDevice) != std::end(devices)) {
      badDevice += 1;
      timeoutCount += 1;
      REQUIRE(timeoutCount < timeout);  // give up after a while
    }

    // Scenario5
    HIP_CHECK_ERROR(hipDeviceGetName(name.data(), name.size(), badDevice), hipErrorInvalidDevice);
  }
}

TEST_CASE("Unit_hipDeviceGetName_CheckPropName") {
  int numDevices = 0;
  std::array<char, LEN> name;
  hipDevice_t device;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(hipDeviceGetName(name.data(), name.size(), device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    // Scenario1
    CHECK(strncmp(name.data(), prop.name, name.size()) == 0);
  }
}

TEST_CASE("Unit_hipDeviceGetName_PartialFill") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-108");
  return;
#endif
  std::array<char, LEN> name;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  auto ordinal = GENERATE_COPY(range(0, numDevices));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, ordinal));
  HIP_CHECK(hipDeviceGetName(name.data(), name.size(), device));

  auto start = std::begin(name);
  auto end = std::end(name);
  const auto len = std::distance(start, std::find(start, end, 0));

  // fill up only half of the length
  const auto fillLen = len / 2;
  constexpr char fillValue = 1;
  std::fill(start, end, fillValue);

  // Scenario6
  HIP_CHECK(hipDeviceGetName(name.data(), fillLen, device));

  const auto strEnd = start + fillLen - 1;
  REQUIRE(std::all_of(start, strEnd, [](char& c) { return c != 0; }));
  REQUIRE(*strEnd == 0);
  REQUIRE(std::all_of(strEnd+1, end, [](char& c) { return c == fillValue; }));
}
