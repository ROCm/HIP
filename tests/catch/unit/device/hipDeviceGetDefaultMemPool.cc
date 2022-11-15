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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

TEST_CASE("Unit_hipDeviceGetDefaultMemPool_Positive_Basic") {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    HipTest::HIP_SKIP_TEST("Test only runs on devices with memory pool support");
    return;
  }

  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, device));
  REQUIRE(mem_pool != nullptr);
}

TEST_CASE("Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") {
  hipMemPool_t mem_pool;

  SECTION("mem_pool == nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetDefaultMemPool(nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("device < 0") {
    HIP_CHECK_ERROR(hipDeviceGetDefaultMemPool(&mem_pool, -1), hipErrorInvalidDevice);
  }

  SECTION("device ordinance too large") {
    HIP_CHECK_ERROR(hipDeviceGetDefaultMemPool(&mem_pool, HipTest::getDeviceCount()),
                    hipErrorInvalidDevice);
  }
}