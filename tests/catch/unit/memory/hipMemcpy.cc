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

TEST_CASE("Unit_hipMemcpy_Basic") { MemcpyWithDirectionCommonTests(hipMemcpy, false); }

TEST_CASE("Unit_hipMemcpy_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  // For transfers from pageable host memory to device memory, a stream sync is performed before
  // the copy is initiated. The function will return once the pageable buffer has been copied to
  // the staging memory for DMA transfer to device memory, but the DMA to final destination may
  // not have completed.
  // For transfers from pinned host memory to device memory, the function is synchronous with
  // respect to the host
  SECTION("Host memory to device memory") {
    MemcpyHtoDSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToDevice), true);
  }

  // For transfers from device to either pageable or pinned host memory, the function returns only
  // once the copy has completed
  SECTION("Device memory to host memory") {
    const auto f = std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToHost);
    MemcpyDtoHPageableSyncBehavior(f, true);
    MemcpyDtoHPinnedSyncBehavior(f, true);
  }

  // For transfers from device memory to device memory, no host-side synchronization is performed.
  SECTION("Device memory to device memory") {
    // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
    // respect to the host
#if HT_AMD
    HipTest::HIP_SKIP_TEST(
        "EXSWCPHIPT-127 - Memcpy from device to device memory behavior differs on AMD and Nvidia");
    return;
#endif
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToDevice), false);
  }

  // For transfers from any host memory to any host memory, the function is fully synchronous with
  // respect to the host
  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToHost), true);
  }
}


TEST_CASE("Unit_hipMemcpy_Negative_Parameters") {
  using namespace std::placeholders;

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToDevice),
                              device_alloc.ptr(), host_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpy(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidMemcpyDirection);
    }
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToHost),
                              host_alloc.ptr(), device_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpy(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidMemcpyDirection);
    }
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToHost),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(
          hipMemcpy(dst_alloc.ptr(), src_alloc.ptr(), kPageSize, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToDevice),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(
          hipMemcpy(dst_alloc.ptr(), src_alloc.ptr(), kPageSize, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
  }
}