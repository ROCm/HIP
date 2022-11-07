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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <array>

#include <hip_test_common.hh>
#include <threaded_zig_zag_test.hh>

namespace {
constexpr std::array<hipSharedMemConfig, 3> kMemConfigs{
    hipSharedMemBankSizeDefault, hipSharedMemBankSizeFourByte, hipSharedMemBankSizeEightByte};
}  // anonymous namespace

TEST_CASE("Unit_hipDeviceSetSharedMemConfig_Positive_Basic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto mem_config = GENERATE(from_range(std::begin(kMemConfigs), std::end(kMemConfigs)));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is " << device);

#if HT_AMD
  HIP_CHECK_ERROR(hipDeviceSetSharedMemConfig(mem_config), hipErrorNotSupported);
#elif HT_NVIDIA
  HIP_CHECK(hipDeviceSetSharedMemConfig(mem_config));
#endif
}

TEST_CASE("Unit_hipDeviceSetSharedMemConfig_Negative_Parameters") {
#if HT_AMD
  HIP_CHECK_ERROR(hipDeviceSetSharedMemConfig(static_cast<hipSharedMemConfig>(-1)),
                  hipErrorNotSupported);
#elif HT_NVIDIA
  HIP_CHECK_ERROR(hipDeviceSetSharedMemConfig(static_cast<hipSharedMemConfig>(-1)),
                  hipErrorInvalidValue);
#endif
}

TEST_CASE("Unit_hipDeviceGetSharedMemConfig_Positive_Default") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is " << device);

  hipSharedMemConfig mem_config;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&mem_config));
  REQUIRE(mem_config == hipSharedMemBankSizeFourByte);
}

TEST_CASE("Unit_hipDeviceGetSharedMemConfig_Positive_Basic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto mem_config = GENERATE(from_range(std::begin(kMemConfigs), std::end(kMemConfigs)));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is " << device);

  HIP_CHECK(hipDeviceSetSharedMemConfig(mem_config));

  hipSharedMemConfig returned_mem_config;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&returned_mem_config));

  int major = -1, minor = -1;
  HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));
  REQUIRE(major > 0);
  if (major == 3 /*Kepler*/) {
    REQUIRE(returned_mem_config == mem_config);
  } else {
    REQUIRE(returned_mem_config == hipSharedMemBankSizeFourByte);
  }
}

TEST_CASE("Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") {
  class HipDeviceGetSharedMemConfigTest
      : public ThreadedZigZagTest<HipDeviceGetSharedMemConfigTest> {
   public:
    HipDeviceGetSharedMemConfigTest(const hipSharedMemConfig mem_config)
        : mem_config_{mem_config} {}

    void TestPart2() { HIP_CHECK_THREAD(hipDeviceSetSharedMemConfig(mem_config_)); }

    void TestPart3() {
      hipSharedMemConfig returned_mem_config;
      HIP_CHECK(hipDeviceGetSharedMemConfig(&returned_mem_config));

      int major = -1, minor = -1;
      HIP_CHECK(hipDeviceComputeCapability(&major, &minor, 0));
      REQUIRE(major > 0);
      if (major == 3 /*Kepler*/) {
        REQUIRE(returned_mem_config == mem_config_);
      } else {
        REQUIRE(returned_mem_config == hipSharedMemBankSizeFourByte);
      }
    }

   private:
    const hipSharedMemConfig mem_config_;
  };

  const auto mem_config = GENERATE(from_range(std::begin(kMemConfigs), std::end(kMemConfigs)));

  HipDeviceGetSharedMemConfigTest test(mem_config);
  test.run();
}

TEST_CASE("Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") {
  HIP_CHECK_ERROR(hipDeviceGetSharedMemConfig(nullptr), hipErrorInvalidValue);
}