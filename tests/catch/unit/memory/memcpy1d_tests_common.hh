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

#pragma once

#include <functional>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

static inline unsigned int GenerateLinearAllocationFlagCombinations(
    const LinearAllocs allocation_type) {
  switch (allocation_type) {
    case LinearAllocs::hipHostMalloc:
      return GENERATE(hipHostMallocDefault, hipHostMallocPortable, hipHostMallocMapped,
                      hipHostMallocWriteCombined);
    case LinearAllocs::mallocAndRegister:
    case LinearAllocs::hipMallocManaged:
    case LinearAllocs::malloc:
    case LinearAllocs::hipMalloc:
      return 0u;
    default:
      assert("Invalid LinearAllocs enumerator");
      throw std::invalid_argument("Invalid LinearAllocs enumerator");
  }
}

template <bool should_synchronize, typename F>
void MemcpyDeviceToHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto host_allocation_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto host_allocation_flags = GenerateLinearAllocationFlagCombinations(host_allocation_type);

  LinearAllocGuard<int> host_allocation(host_allocation_type, allocation_size,
                                        host_allocation_flags);
  LinearAllocGuard<int> device_allocation(LA::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*device_allocation.ptr());
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 42;
  VectorSet<<<block_count, thread_count, 0, kernel_stream>>>(device_allocation.ptr(),
                                                             expected_value, element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(memcpy_func(host_allocation.host_ptr(), device_allocation.ptr(), allocation_size));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  ArrayFindIfNot(host_allocation.host_ptr(), expected_value, element_count);
}

template <bool should_synchronize, typename F>
void MemcpyHostToDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto host_allocation_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto host_allocation_flags = GenerateLinearAllocationFlagCombinations(host_allocation_type);

  LinearAllocGuard<int> src_host_allocation(host_allocation_type, allocation_size,
                                            host_allocation_flags);
  LinearAllocGuard<int> dst_host_allocation(LA::hipHostMalloc, allocation_size);
  LinearAllocGuard<int> device_allocation(LA::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*device_allocation.ptr());
  constexpr int fill_value = 42;
  std::fill_n(src_host_allocation.host_ptr(), element_count, fill_value);
  std::fill_n(dst_host_allocation.host_ptr(), element_count, 0);

  HIP_CHECK(memcpy_func(device_allocation.ptr(), src_host_allocation.host_ptr(), allocation_size));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(hipMemcpy(dst_host_allocation.host_ptr(), device_allocation.ptr(), allocation_size,
                      hipMemcpyDeviceToHost));

  ArrayFindIfNot(dst_host_allocation.host_ptr(), fill_value, element_count);
}

template <bool should_synchronize, typename F>
void MemcpyHostToHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto src_allocation_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto dst_allocation_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto src_allocation_flags = GenerateLinearAllocationFlagCombinations(src_allocation_type);
  const auto dst_allocation_flags = GenerateLinearAllocationFlagCombinations(dst_allocation_type);

  LinearAllocGuard<int> src_allocation(src_allocation_type, allocation_size, src_allocation_flags);
  LinearAllocGuard<int> dst_allocation(dst_allocation_type, allocation_size, dst_allocation_flags);

  const auto element_count = allocation_size / sizeof(*src_allocation.host_ptr());
  constexpr auto expected_value = 42;
  std::fill_n(src_allocation.host_ptr(), element_count, expected_value);

  HIP_CHECK(memcpy_func(dst_allocation.host_ptr(), src_allocation.host_ptr(), allocation_size));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  ArrayFindIfNot(dst_allocation.host_ptr(), expected_value, element_count);
}

template <bool should_synchronize, bool enable_peer_access, typename F>
void MemcpyDeviceToDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto device_count = HipTest::getDeviceCount();
  const auto src_device = GENERATE_COPY(range(0, device_count));
  const auto dst_device = GENERATE_COPY(range(0, device_count));
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  HIP_CHECK(hipSetDevice(src_device));
  if constexpr (enable_peer_access) {
    if (src_device == dst_device) {
      return;
    }
    int can_access_peer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
    if (!can_access_peer) {
      INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
      REQUIRE(can_access_peer);
    }
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));
  }

  LinearAllocGuard<int> src_allocation(LinearAllocs::hipMalloc, allocation_size);
  LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, allocation_size, hipHostMallocPortable);
  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocGuard<int> dst_allocation(LinearAllocs::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*src_allocation.ptr());
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 42;
  HIP_CHECK(hipSetDevice(src_device));
  VectorSet<<<block_count, thread_count, 0, kernel_stream>>>(src_allocation.ptr(), expected_value,
                                                             element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(memcpy_func(dst_allocation.ptr(), src_allocation.ptr(), allocation_size));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_allocation.ptr(), allocation_size, hipMemcpyDeviceToHost));
  if constexpr (enable_peer_access) {
    // If we've gotten this far, EnablePeerAccess must have succeeded, so we only need to check this
    // condition
    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  }

  ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
}

template <bool should_synchronize, typename F> void MemcpyWithDirectionCommonTests(F memcpy_func) {
  using namespace std::placeholders;
  SECTION("Device to host") {
    MemcpyDeviceToHostShell<should_synchronize>(
        std::bind(memcpy_func, _1, _2, _3, hipMemcpyDeviceToHost));
  }

  SECTION("Device to host with default kind") {
    MemcpyDeviceToHostShell<should_synchronize>(
        std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault));
  }

  SECTION("Host to device") {
    MemcpyHostToDeviceShell<should_synchronize>(
        std::bind(memcpy_func, _1, _2, _3, hipMemcpyHostToDevice));
  }

  SECTION("Host to device with default kind") {
    MemcpyHostToDeviceShell<should_synchronize>(
        std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault));
  }

  SECTION("Host to host") {
    MemcpyHostToHostShell<should_synchronize>(
        std::bind(memcpy_func, _1, _2, _3, hipMemcpyHostToHost));
  }

  SECTION("Host to host with default kind") {
    MemcpyHostToHostShell<should_synchronize>(std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault));
  }

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<should_synchronize, true>(
          std::bind(memcpy_func, _1, _2, _3, hipMemcpyDeviceToDevice));
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<should_synchronize, false>(
          std::bind(memcpy_func, _1, _2, _3, hipMemcpyDeviceToDevice));
    }
  }

  SECTION("Device to device with default kind") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<should_synchronize, true>(
          std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault));
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<should_synchronize, false>(
          std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault));
    }
  }
}

// Synchronization behavior checks
template <typename F>
void MemcpySyncBehaviorCheck(F memcpy_func, const bool should_sync,
                             const hipStream_t kernel_stream) {
  LaunchDelayKernel(std::chrono::milliseconds{100}, kernel_stream);
  HIP_CHECK(memcpy_func());
  if (should_sync) {
    HIP_CHECK(hipStreamQuery(kernel_stream));
  } else {
    HIP_CHECK_ERROR(hipStreamQuery(kernel_stream), hipErrorNotReady);
  }
}

template <typename F>
void MemcpyHtoDSyncBehavior(F memcpy_func, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto host_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  LinearAllocGuard<int> host_alloc(host_alloc_type, kPageSize);
  LinearAllocGuard<int> device_alloc(LA::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, device_alloc.ptr(), host_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyDtoHPageableSyncBehavior(F memcpy_func, const bool should_sync,
                                    const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, kPageSize);
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), device_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyDtoHPinnedSyncBehavior(F memcpy_func, const bool should_sync,
                                  const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), device_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyDtoDSyncBehavior(F memcpy_func, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, dst_alloc.ptr(), src_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyHtoHSyncBehavior(F memcpy_func, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto src_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto dst_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);

  LinearAllocGuard<int> src_alloc(src_alloc_type, kPageSize);
  LinearAllocGuard<int> dst_alloc(dst_alloc_type, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, dst_alloc.ptr(), src_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

// Common negative tests
template <typename F> void MemcpyCommonNegativeTests(F f, void* dst, void* src, size_t count) {
  SECTION("dst == nullptr") { HIP_CHECK_ERROR(f(nullptr, src, count), hipErrorInvalidValue); }
  SECTION("src == nullptr") { HIP_CHECK_ERROR(f(dst, nullptr, count), hipErrorInvalidValue); }
}

template <typename F>
void MemcpyWithDirectionCommonNegativeTests(F f, void* dst, void* src, size_t count,
                                            hipMemcpyKind kind) {
  using namespace std::placeholders;
  MemcpyCommonNegativeTests(std::bind(f, _1, _2, _3, kind), dst, src, count);

// Disabled on AMD due to defect - EXSWHTEC-128
#if HT_NVIDIA
  SECTION("Invalid MemcpyKind") {
    HIP_CHECK_ERROR(f(dst, src, count, static_cast<hipMemcpyKind>(-1)),
                    hipErrorInvalidMemcpyDirection);
  }
#endif
}