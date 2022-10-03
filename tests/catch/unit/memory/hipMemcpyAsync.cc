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

TEST_CASE("Unit_hipMemcpyAsync_Basic") {
  using namespace std::placeholders;
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  MemcpyWithDirectionCommonTests(std::bind(hipMemcpyAsync, _1, _2, _3, _4, stream), true);
}


TEST_CASE("Unit_hipMemcpyAsync_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host memory to device memory") {
    // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
    // respect to the host
#if HT_AMD
    HipTest::HIP_SKIP_TEST(
        "EXSWCPHIPT-127 - MemcpyAsync from host to device memory behavior differs on AMD and "
        "Nvidia");
    return;
#endif
    MemcpyHtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
                           false);
  }

  SECTION("Device memory to pageable host memory") {
    MemcpyDtoHPageableSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), true);
  }

  SECTION("Device memory to pinned host memory") {
    MemcpyDtoHPinnedSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), false);
  }

  SECTION("Device memory to device memory") {
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr),
                           false);
  }

  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                           true);
  }
}

TEST_CASE("Unit_hipMemcpyAsync_Negative_Parameters") {
  using namespace std::placeholders;

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
                              device_alloc.ptr(), host_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyAsync(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                     hipMemcpyHostToDevice, stream),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr),
                              host_alloc.ptr(), device_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyAsync(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                     hipMemcpyDeviceToHost, stream),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(
          hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize, hipMemcpyHostToHost, stream),
          hipErrorInvalidValue);
    }
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr), dst_alloc.ptr(),
        src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(src_alloc.ptr(), dst_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     hipMemcpyDeviceToDevice, stream),
                      hipErrorInvalidValue);
    }
  }
}