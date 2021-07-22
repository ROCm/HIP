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
#include <hip_test_common.hh>
/**
 * hipDeviceGetCacheConfig tests
 * Scenario1: Validates if pConfig = nullptr returns hip error code.
 * Scenario2: Validates if the value returned by hipDeviceGetCacheConfig is valid.
 */
TEST_CASE("Unit_hipDeviceGetCacheConfig_NegTst") {
  // Scenario1
  REQUIRE_FALSE(hipSuccess == hipDeviceGetCacheConfig(nullptr));
}

TEST_CASE("Unit_hipDeviceGetCacheConfig_FuncTst") {
  hipFuncCache_t cacheConfig;
  // Scenario2
  HIP_CHECK(hipDeviceGetCacheConfig(&cacheConfig));
  REQUIRE_FALSE(((cacheConfig != hipFuncCachePreferNone) &&
                (cacheConfig != hipFuncCachePreferShared) &&
                (cacheConfig != hipFuncCachePreferL1) &&
                (cacheConfig != hipFuncCachePreferEqual)));
  // This code exists to test the dummy implementation of
  // hipDeviceSetCacheConfig.
  HIP_CHECK(hipDeviceSetCacheConfig(cacheConfig));
}
