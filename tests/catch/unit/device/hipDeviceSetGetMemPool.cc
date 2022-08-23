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

TEST_CASE("Unit_hipDeviceSetMemPool_Positive_Basic") {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    HipTest::HIP_SKIP_TEST("Test only runs on devices with memory pool support");
    return;
  }

#if HT_NVIDIA
  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, device));
  HIP_CHECK(hipDeviceSetMemPool(device, mem_pool));
#endif

#if HT_AMD
// TODO
#endif
}

TEST_CASE("Unit_hipDeviceGetMemPool_Positive_Basic") {
  const int device = GENERATE(range(0, HipTest::getDeviceCount()));

  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    HipTest::HIP_SKIP_TEST("Test only runs on devices with memory pool support");
    return;
  }

#if HT_NVIDIA
  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetMemPool(&mem_pool, device));
  REQUIRE(mem_pool != nullptr);
#endif

#if HT_AMD
// TODO
// Mempool management APIs seem to still be unsupported on NVIDIA platforms in ROCm 5.2.3 despite
// appearances
#endif
}

TEST_CASE("Unit_hipDeviceSetMemPool_Negative_Parameters") {
  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, 0));

  SECTION("mem_pool == nullptr") {
    HIP_CHECK_ERROR(hipDeviceSetMemPool(0, nullptr), hipErrorInvalidValue);
  }

  SECTION("device < 0") {
    HIP_CHECK_ERROR(hipDeviceSetMemPool(-1, mem_pool), hipErrorInvalidValue);
  }

  SECTION("device ordinance too large") {
    HIP_CHECK_ERROR(hipDeviceSetMemPool(HipTest::getDeviceCount(), mem_pool), hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipDeviceGetMemPool_Negative_Parameters") {
  hipMemPool_t mem_pool;

  SECTION("mem_pool == nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetMemPool(nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("device < 0") {
    HIP_CHECK_ERROR(hipDeviceGetMemPool(&mem_pool, -1), hipErrorInvalidValue);
  }

  SECTION("device ordinance too large") {
    HIP_CHECK_ERROR(hipDeviceGetMemPool(&mem_pool, HipTest::getDeviceCount()),
                    hipErrorInvalidValue);
  }
}