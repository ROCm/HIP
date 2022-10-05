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
constexpr std::array<hipFuncCache_t, 4> kCacheConfigs{
    hipFuncCachePreferNone, hipFuncCachePreferShared, hipFuncCachePreferL1,
    hipFuncCachePreferEqual};
}  // anonymous namespace


TEST_CASE("Unit_hipDeviceSetCacheConfig_Positive_Basic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  const auto cache_config =
      GENERATE(from_range(std::begin(kCacheConfigs), std::end(kCacheConfigs)));
#if HT_AMD
  HIP_CHECK_ERROR(hipDeviceSetCacheConfig(cache_config), hipErrorNotSupported);
#elif HT_NVIDIA
  HIP_CHECK(hipDeviceSetCacheConfig(cache_config));
#endif
}

TEST_CASE("Unit_hipDeviceSetCacheConfig_Negative_Parameters") {
#if HT_AMD
  HIP_CHECK_ERROR(hipDeviceSetCacheConfig(static_cast<hipFuncCache_t>(-1)), hipErrorNotSupported);
#elif HT_NVIDIA
  HIP_CHECK_ERROR(hipDeviceSetCacheConfig(static_cast<hipFuncCache_t>(-1)), hipErrorInvalidValue);
#endif
}

TEST_CASE("Unit_hipDeviceGetCacheConfig_Positive_Default") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  hipFuncCache_t cache_config;
  HIP_CHECK(hipDeviceGetCacheConfig(&cache_config));
  REQUIRE(cache_config == hipFuncCachePreferNone);
}

TEST_CASE("Unit_hipDeviceGetCacheConfig_Positive_Basic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  const auto cache_config =
      GENERATE(from_range(std::begin(kCacheConfigs), std::end(kCacheConfigs)));

  HIP_CHECK(hipDeviceSetCacheConfig(cache_config));
  hipFuncCache_t returned_cache_config;
  HIP_CHECK(hipDeviceGetCacheConfig(&returned_cache_config));

  REQUIRE(returned_cache_config == cache_config);
}

TEST_CASE("Unit_hipDeviceGetCacheConfig_Positive_Threaded") {
  class HipDeviceSetGetCacheConfigTest : public ThreadedZigZagTest<HipDeviceSetGetCacheConfigTest> {
   public:
    HipDeviceSetGetCacheConfigTest(const hipFuncCache_t cache_config)
        : cache_config_{cache_config} {}

    void TestPart2() { HIP_CHECK_THREAD(hipDeviceSetCacheConfig(cache_config_)); }

    void TestPart3() {
      hipFuncCache_t returned_cache_config;
      HIP_CHECK(hipDeviceGetCacheConfig(&returned_cache_config));
      REQUIRE(returned_cache_config == cache_config_);
    }

   private:
    const hipFuncCache_t cache_config_;
  };

  const auto cache_config =
      GENERATE(from_range(std::begin(kCacheConfigs), std::end(kCacheConfigs)));

  HipDeviceSetGetCacheConfigTest test(cache_config);
  test.run();
}

TEST_CASE("Unit_HipDeviceGetCacheConfig_Negative_Parameters") {
  HIP_CHECK_ERROR(hipDeviceGetCacheConfig(nullptr), hipErrorInvalidValue);
}