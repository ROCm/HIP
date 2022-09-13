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

TEST_CASE("Unit_hipDeviceReset_Positive_Basic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  unsigned int flags_before = 0u;
  HIP_CHECK(hipGetDeviceFlags(&flags_before));
  hipSharedMemConfig mem_config_before;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&mem_config_before));

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 500));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipDeviceSetCacheConfig(hipFuncCachePreferL1));
  HIP_CHECK(hipDeviceSetSharedMemConfig(mem_config_before == hipSharedMemBankSizeFourByte
                                            ? hipSharedMemBankSizeEightByte
                                            : hipSharedMemBankSizeFourByte));
  HIP_CHECK(hipSetDeviceFlags(flags_before ^ (1u << 2)));

  HIP_CHECK(hipDeviceReset());

  CHECK(hipFree(ptr) == hipErrorInvalidValue);

  CHECK(hipStreamDestroy(stream) == hipErrorContextIsDestroyed);

  unsigned int flags_after = 0u;
  CHECK(hipGetDeviceFlags(&flags_after) == hipSuccess);
  CHECK(flags_after == flags_before);

  hipFuncCache_t cache_config;
  CHECK(hipDeviceGetCacheConfig(&cache_config) == hipSuccess);
  CHECK(cache_config == hipFuncCachePreferNone);

  hipSharedMemConfig mem_config_after;
  CHECK(hipDeviceGetSharedMemConfig(&mem_config_after) == hipSuccess);
  CHECK(mem_config_after == mem_config_before);
}

TEST_CASE("Unit_hipDeviceReset_Positive_Threaded") {
  HIP_CHECK(hipSetDevice(0));
  INFO("Current device is: " << 0);

  unsigned int flags_before = 0u;
  HIP_CHECK(hipGetDeviceFlags(&flags_before));
  hipSharedMemConfig mem_config_before;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&mem_config_before));

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 500));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipDeviceSetCacheConfig(hipFuncCachePreferL1));
  HIP_CHECK(hipDeviceSetSharedMemConfig(mem_config_before == hipSharedMemBankSizeFourByte
                                            ? hipSharedMemBankSizeEightByte
                                            : hipSharedMemBankSizeFourByte));
  HIP_CHECK(hipSetDeviceFlags(flags_before ^ (1u << 2)));

  std::thread([] {
    HIP_CHECK_THREAD(hipSetDevice(0));
    HIP_CHECK_THREAD(hipDeviceReset());
  }).join();
  HIP_CHECK_THREAD_FINALIZE();

  CHECK(hipFree(ptr) == hipErrorInvalidValue);

  CHECK(hipStreamDestroy(stream) == hipErrorContextIsDestroyed);

  unsigned int flags_after = 0u;
  CHECK(hipGetDeviceFlags(&flags_after) == hipSuccess);
  CHECK(flags_after == flags_before);

  hipFuncCache_t cache_config;
  CHECK(hipDeviceGetCacheConfig(&cache_config) == hipSuccess);
  CHECK(cache_config == hipFuncCachePreferNone);

  hipSharedMemConfig mem_config_after;
  CHECK(hipDeviceGetSharedMemConfig(&mem_config_after) == hipSuccess);
  CHECK(mem_config_after == mem_config_before);
}