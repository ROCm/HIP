/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Verifies functionality of hipSetDevice/hipGetDevice api.
 * -- Basic Test to set and get valid device numbers.
 */

#include <hip_test_common.hh>
#include <thread>

TEST_CASE("Unit_hipSetDevice_BasicSetGet") {
  int numDevices = 0;
  int device{};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices != 0);

  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipGetDevice(&device));
    REQUIRE(device == i);

    // Check for hipDevice_t as well
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, i));
  }
}

TEST_CASE("Unit_hipGetSetDevice_MultiThreaded") {
  auto maxThreads = std::thread::hardware_concurrency();
  auto deviceCount = HipTest::getDeviceCount();

  auto thread = [&]() {
    for (int i = 0; i < deviceCount; i++) {
      HIP_CHECK_THREAD(hipSetDevice(i));
      int get = -1;
      HIP_CHECK_THREAD(hipGetDevice(&get));
      REQUIRE_THREAD(get == i);

      // check hipDeviceGet
      hipDevice_t device;
      HIP_CHECK_THREAD(hipDeviceGet(&device, i));

      // Alloc some memory and set it
      unsigned int* ptr{nullptr};
      HIP_CHECK_THREAD(hipMalloc(&ptr, sizeof(unsigned int)));
      REQUIRE_THREAD(ptr != nullptr);
      HIP_CHECK_THREAD(hipMemset(ptr, 0x0A, sizeof(unsigned int)));
      int res{0};
      HIP_CHECK_THREAD(hipMemcpy(&res, ptr, sizeof(unsigned int), hipMemcpyDeviceToHost));
      REQUIRE_THREAD(res == 0x0A0A0A0A);
      HIP_CHECK_THREAD(hipFree(ptr));
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(maxThreads);

  for (int i = 0; i < maxThreads; i++) {
    pool.emplace_back(std::thread(thread));
  }

  for (auto& i : pool) {
    i.join();
  }

  HIP_CHECK_THREAD_FINALIZE();
}

TEST_CASE("Unit_hipSetGetDevice_Negative") {
  SECTION("Get Device - nullptr") { HIP_CHECK_ERROR(hipGetDevice(nullptr), hipErrorInvalidValue); }

  SECTION("Set Device - -1") { HIP_CHECK_ERROR(hipSetDevice(-1), hipErrorInvalidDevice); }

  SECTION("Set Device - NumDevices + 1") {
    HIP_CHECK_ERROR(hipSetDevice(HipTest::getDeviceCount()), hipErrorInvalidDevice);
  }
}

TEST_CASE("Unit_hipDeviceGet_Negative") {
  // TODO enable after EXSWCPHIPT-104 is fixed
#if HT_NVIDIA
  SECTION("Nullptr as handle") { HIP_CHECK_ERROR(hipDeviceGet(nullptr, 0), hipErrorInvalidValue); }
#endif

  SECTION("Out of bound ordial - positive") {
    hipDevice_t device{};
    auto totalDevices = HipTest::getDeviceCount();
    HIP_CHECK_ERROR(hipDeviceGet(&device, totalDevices), hipErrorInvalidDevice);
  }

  SECTION("Out of bound ordial - negative") {
    hipDevice_t device{};
    HIP_CHECK_ERROR(hipDeviceGet(&device, -1), hipErrorInvalidDevice);
  }
}
