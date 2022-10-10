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

#include "linear_memcpy_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

// hipMemcpyDtoH
TEST_CASE("Unit_hipMemcpyDtoH_Basic") {
  MemcpyDeviceToHostShell<false>([](void* dst, void* src, size_t count) {
    return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
  });
}

TEST_CASE("Unit_hipMemcpyDtoH_Synchronization_Behavior") {
  const auto f = [](void* dst, void* src, size_t count) {
    return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
  };
  MemcpyDtoHPageableSyncBehavior(f, true);
  MemcpyDtoHPinnedSyncBehavior(f, true);
}

TEST_CASE("Unit_hipMemcpyDtoH_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      host_alloc.ptr(), device_alloc.ptr(), kPageSize);
}

// hipMemcpyHtoD
TEST_CASE("Unit_hipMemcpyHtoD_Basic") {
  MemcpyHostToDeviceShell<false>([](void* dst, void* src, size_t count) {
    return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
  });
}

TEST_CASE("Unit_hipMemcpyHtoD_Synchronization_Behavior") {
  MemcpyHtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      true);
}

TEST_CASE("Unit_hipMemcpyHtoD_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      device_alloc.ptr(), host_alloc.ptr(), kPageSize);
}

// hipMemcpyDtoD
TEST_CASE("Unit_hipMemcpyDtoD_Basic") {
  const auto f = [](void* dst, void* src, size_t count) {
    return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                         reinterpret_cast<hipDeviceptr_t>(src), count);
  };
  SECTION("Peer access enabled") { MemcpyDeviceToDeviceShell<false, true>(f); }
  SECTION("Peer access disabled") { MemcpyDeviceToDeviceShell<false, false>(f); }
}

TEST_CASE("Unit_hipMemcpyDtoD_Synchronization_Behavior") {
  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-127 - Memcpy from device to device memory behavior differs on AMD and Nvidia");
  return;
#endif
  MemcpyDtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyDtoD_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      dst_alloc.ptr(), src_alloc.ptr(), kPageSize);
}