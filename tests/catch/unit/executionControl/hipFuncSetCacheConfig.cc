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

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

namespace {
constexpr std::array<hipFuncCache_t, 4> kCacheConfigs{
    hipFuncCachePreferNone, hipFuncCachePreferShared, hipFuncCachePreferL1,
    hipFuncCachePreferEqual};
}  // anonymous namespace

TEST_CASE("Unit_hipFuncSetCacheConfig_Positive_Basic") {
  const auto cache_config = GENERATE(from_range(begin(kCacheConfigs), end(kCacheConfigs)));

  HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<void*>(kernel), cache_config));

  kernel<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());
}

TEST_CASE("Unit_hipFuncSetCacheConfig_Negative_Parameters") {
  SECTION("func == nullptr") {
    HIP_CHECK_ERROR(hipFuncSetCacheConfig(nullptr, hipFuncCachePreferNone),
                    hipErrorInvalidDeviceFunction);
  }
  SECTION("invalid cache config") {
    HIP_CHECK_ERROR(
        hipFuncSetCacheConfig(reinterpret_cast<void*>(kernel), static_cast<hipFuncCache_t>(-1)),
        hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipFuncSetCacheConfig_Negative_Not_Supported") {
  HIP_CHECK_ERROR(hipFuncSetCacheConfig(reinterpret_cast<void*>(kernel), hipFuncCachePreferNone),
                  hipErrorNotSupported);
}