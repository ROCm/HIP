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
#include "hipArrayCommon.hh"

/* Array -> Host */
template <bool should_synchronize, typename T, typename F>
void MemcpyAtoHShell(F memcpy_func, size_t width, const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;

  size_t allocation_size = width * sizeof(T);

  LinearAllocGuard<T> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<T> array_allocation(make_hipExtent(width, 0, 0), flag);

  const auto element_count = allocation_size / sizeof(T);
  constexpr int fill_value = 42;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  HIP_CHECK(hipMemcpy2DToArray(array_allocation.ptr(), 0, 0, host_allocation.host_ptr(),
                               sizeof(T) * width, sizeof(T) * width, 1, hipMemcpyHostToDevice));
  std::fill_n(host_allocation.host_ptr(), element_count, 0);

  HIP_CHECK(memcpy_func(host_allocation.host_ptr(), array_allocation.ptr()));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  ArrayFindIfNot(host_allocation.host_ptr(), fill_value, element_count);
}

template <bool should_synchronize, typename T, typename F>
void Memcpy2DHostFromAShell(F memcpy_func, size_t width, size_t height,
                            const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;

  size_t allocation_size = width * height * sizeof(T);

  LinearAllocGuard<T> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<T> array_allocation(make_hipExtent(width, height, 0), flag);

  const auto element_count = allocation_size / sizeof(T);
  constexpr int fill_value = 42;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  HIP_CHECK(hipMemcpy2DToArray(array_allocation.ptr(), 0, 0, host_allocation.host_ptr(),
                               sizeof(T) * width, sizeof(T) * width, height,
                               hipMemcpyHostToDevice));
  std::fill_n(host_allocation.host_ptr(), element_count, 0);

  HIP_CHECK(memcpy_func(host_allocation.host_ptr(), sizeof(T) * width, array_allocation.ptr()));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  ArrayFindIfNot(host_allocation.host_ptr(), fill_value, element_count);
}

/* Array -> Device */
template <bool should_synchronize, bool enable_peer_access, typename T, typename F>
void Memcpy2DDeviceFromAShell(F memcpy_func, size_t width, size_t height,
                              const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  size_t allocation_size = width * height * sizeof(T);

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

  LinearAllocGuard<T> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<T> array_allocation(make_hipExtent(width, height, 0), flag);
  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocGuard2D<T> device_allocation(width, height);

  HIP_CHECK(hipSetDevice(src_device));
  const auto element_count = allocation_size / sizeof(T);
  constexpr int fill_value = 42;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  HIP_CHECK(hipMemcpy2DToArray(array_allocation.ptr(), 0, 0, host_allocation.host_ptr(),
                               sizeof(T) * width, sizeof(T) * width, height,
                               hipMemcpyHostToDevice));
  std::fill_n(host_allocation.host_ptr(), element_count, 0);

  HIP_CHECK(
      memcpy_func(device_allocation.ptr(), device_allocation.pitch(), array_allocation.ptr()));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(hipMemcpy2D(host_allocation.host_ptr(), sizeof(T) * width, device_allocation.ptr(),
                        device_allocation.pitch(), sizeof(T) * width, height,
                        hipMemcpyDeviceToHost));

  if constexpr (enable_peer_access) {
    // If we've gotten this far, EnablePeerAccess must have succeeded, so we only need to check this
    // condition
    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  }

  ArrayFindIfNot(host_allocation.host_ptr(), fill_value, element_count);
}

/* Host -> Array */
template <bool should_synchronize, typename T, typename F>
void MemcpyHtoAShell(F memcpy_func, size_t width, const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;

  size_t allocation_size = width * sizeof(T);

  LinearAllocGuard<T> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<T> array_allocation(make_hipExtent(width, 0, 0), flag);

  const auto element_count = allocation_size / sizeof(T);
  constexpr int fill_value = 41;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  HIP_CHECK(memcpy_func(array_allocation.ptr(), host_allocation.host_ptr()));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  std::fill_n(host_allocation.host_ptr(), element_count, 0);

  HIP_CHECK(hipMemcpy2DFromArray(host_allocation.host_ptr(), sizeof(T) * width,
                                 array_allocation.ptr(), 0, 0, sizeof(T) * width, 1,
                                 hipMemcpyDeviceToHost));

  ArrayFindIfNot(host_allocation.host_ptr(), fill_value, element_count);
}

template <bool should_synchronize, typename T, typename F>
void Memcpy2DHosttoAShell(F memcpy_func, size_t width, size_t height,
                          const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  ;

  size_t allocation_size = width * height * sizeof(T);

  LinearAllocGuard<T> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<T> array_allocation(make_hipExtent(width, height, 0), flag);

  const auto element_count = allocation_size / sizeof(T);
  constexpr int fill_value = 41;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  HIP_CHECK(memcpy_func(array_allocation.ptr(), host_allocation.host_ptr(), sizeof(T) * width));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  std::fill_n(host_allocation.host_ptr(), element_count, 0);

  HIP_CHECK(hipMemcpy2DFromArray(host_allocation.host_ptr(), sizeof(T) * width,
                                 array_allocation.ptr(), 0, 0, sizeof(T) * width, height,
                                 hipMemcpyDeviceToHost));

  ArrayFindIfNot(host_allocation.host_ptr(), fill_value, element_count);
}

/* Device -> Array */
template <bool should_synchronize, bool enable_peer_access, typename T, typename F>
void Memcpy2DDevicetoAShell(F memcpy_func, size_t width, size_t height,
                            const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  size_t allocation_size = width * height * sizeof(T);

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

  LinearAllocGuard<T> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard2D<T> device_allocation(width, height);
  HIP_CHECK(hipSetDevice(dst_device));
  ArrayAllocGuard<T> array_allocation(make_hipExtent(width, height, 0), flag);

  HIP_CHECK(hipSetDevice(src_device));
  const auto element_count = allocation_size / sizeof(T);
  constexpr int fill_value = 41;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  HIP_CHECK(hipMemcpy2D(device_allocation.ptr(), device_allocation.pitch(),
                        host_allocation.host_ptr(), sizeof(T) * width, sizeof(T) * width, height,
                        hipMemcpyHostToDevice));

  HIP_CHECK(
      memcpy_func(array_allocation.ptr(), device_allocation.ptr(), device_allocation.pitch()));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  std::fill_n(host_allocation.host_ptr(), element_count, 0);

  HIP_CHECK(hipMemcpy2DFromArray(host_allocation.host_ptr(), sizeof(T) * width,
                                 array_allocation.ptr(), 0, 0, sizeof(T) * width, height,
                                 hipMemcpyDeviceToHost));

  if constexpr (enable_peer_access) {
    // If we've gotten this far, EnablePeerAccess must have succeeded, so we only need to check this
    // condition
    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  }

  ArrayFindIfNot(host_allocation.host_ptr(), fill_value, element_count);
}

// Synchronization behavior checks
template <typename F>
void MemcpyArraySyncBehaviorCheck(F memcpy_func, const bool should_sync,
                                  const hipStream_t kernel_stream) {
  LaunchDelayKernel(std::chrono::milliseconds{100}, kernel_stream);
  HIP_CHECK(memcpy_func());
  if (should_sync) {
    HIP_CHECK(hipStreamQuery(kernel_stream));
  } else {
    HIP_CHECK_ERROR(hipStreamQuery(kernel_stream), hipErrorNotReady);
  }
}

/* Host -> Array Sync check */
template <typename F>
void MemcpyHtoASyncBehavior(F memcpy_func, size_t width, size_t height, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  size_t num_h = (height == 0) ? 1 : height;
  size_t allocation_size = width * num_h * sizeof(int);

  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), flag);
  MemcpyArraySyncBehaviorCheck(std::bind(memcpy_func, array_allocation.ptr(), host_alloc.ptr()),
                               should_sync, kernel_stream);
}

/* Array -> Host sync check */
template <typename F>
void MemcpyAtoHPageableSyncBehavior(F memcpy_func, size_t width, size_t height,
                                    const bool should_sync,
                                    const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  size_t num_h = (height == 0) ? 1 : height;
  size_t allocation_size = width * num_h * sizeof(int);

  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), flag);
  MemcpyArraySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), array_allocation.ptr()),
                               should_sync, kernel_stream);
}

template <typename F>
void MemcpyAtoHPinnedSyncBehavior(F memcpy_func, size_t width, size_t height,
                                  const bool should_sync,
                                  const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  size_t num_h = (height == 0) ? 1 : height;
  size_t allocation_size = width * num_h * sizeof(int);

  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), flag);
  MemcpyArraySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), array_allocation.ptr()),
                               should_sync, kernel_stream);
}

/* Device -> Array sync check */
template <typename F>
void MemcpyDtoASyncBehavior(F memcpy_func, size_t width, size_t height, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;

  ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), flag);
  LinearAllocGuard2D<int> device_allocation(width, height);

  MemcpyArraySyncBehaviorCheck(std::bind(memcpy_func, array_allocation.ptr(),
                                         device_allocation.ptr(), device_allocation.pitch()),
                               should_sync, kernel_stream);
}

/* Array -> Device sync check */
template <typename F>
void MemcpyAtoDSyncBehavior(F memcpy_func, size_t width, size_t height, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  const unsigned int flag = hipArrayDefault;

  LinearAllocGuard2D<int> device_allocation(width, height);
  ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), flag);

  MemcpyArraySyncBehaviorCheck(std::bind(memcpy_func, device_allocation.ptr(),
                                         device_allocation.pitch(), array_allocation.ptr()),
                               should_sync, kernel_stream);
}

/* Array -> Host/Device zero copy */
template <bool should_synchronize, typename F>
void Memcpy2DFromArrayZeroWidthHeight(F memcpy_func, size_t width, size_t height,
                                      const hipStream_t stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  const auto element_count = width * height;

  SECTION("Device to Host") {
    ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, width * height * sizeof(int));
    int fill_value = 42;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);
    HIP_CHECK(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.host_ptr(),
                                 sizeof(int) * width, sizeof(int) * width, height,
                                 hipMemcpyHostToDevice));
    fill_value = 41;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);

    HIP_CHECK(memcpy_func(host_alloc.host_ptr(), sizeof(int) * width, array_alloc.ptr()));
    if (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);
  }
  SECTION("Device to Device") {
    ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
    LinearAllocGuard2D<int> device_alloc(width, height);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, width * height * sizeof(int));
    int fill_value = 42;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);
    HIP_CHECK(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.host_ptr(),
                                 sizeof(int) * width, sizeof(int) * width, height,
                                 hipMemcpyHostToDevice));
    fill_value = 41;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);
    HIP_CHECK(hipMemcpy2D(device_alloc.ptr(), device_alloc.pitch(), host_alloc.host_ptr(),
                          sizeof(int) * width, sizeof(int) * width, height, hipMemcpyHostToDevice));

    HIP_CHECK(memcpy_func(device_alloc.ptr(), device_alloc.pitch(), array_alloc.ptr()));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipMemcpy2D(host_alloc.host_ptr(), sizeof(int) * width, device_alloc.ptr(),
                          device_alloc.pitch(), sizeof(int) * width, height,
                          hipMemcpyDeviceToHost));
    ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);
  }
}

/* Host/Device -> Array zero copy */
template <bool should_synchronize, typename F>
void Memcpy2DToArrayZeroWidthHeight(F memcpy_func, size_t width, size_t height,
                                    const hipStream_t stream = nullptr) {
  const unsigned int flag = hipArrayDefault;
  const auto element_count = width * height;

  SECTION("Host to Device") {
    ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, width * height * sizeof(int));
    int fill_value = 42;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);
    HIP_CHECK(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.host_ptr(),
                                 sizeof(int) * width, sizeof(int) * width, height,
                                 hipMemcpyHostToDevice));
    fill_value = 41;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);

    HIP_CHECK(memcpy_func(array_alloc.ptr(), host_alloc.host_ptr(), sizeof(int) * width));
    if (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipMemcpy2DFromArray(host_alloc.host_ptr(), sizeof(int) * width, array_alloc.ptr(), 0,
                                   0, sizeof(int) * width, height, hipMemcpyDeviceToHost));
    ArrayFindIfNot(host_alloc.host_ptr(), 42, element_count);
  }
  SECTION("Device to Device") {
    ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
    LinearAllocGuard2D<int> device_alloc(width, height);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, width * height * sizeof(int));
    int fill_value = 42;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);
    HIP_CHECK(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.host_ptr(),
                                 sizeof(int) * width, sizeof(int) * width, height,
                                 hipMemcpyHostToDevice));
    fill_value = 41;
    std::fill_n(host_alloc.host_ptr(), width * height, fill_value);
    HIP_CHECK(hipMemcpy2D(device_alloc.ptr(), device_alloc.pitch(), host_alloc.host_ptr(),
                          sizeof(int) * width, sizeof(int) * width, height, hipMemcpyHostToDevice));

    HIP_CHECK(memcpy_func(array_alloc.ptr(), device_alloc.ptr(), device_alloc.pitch()));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipMemcpy2DFromArray(host_alloc.host_ptr(), sizeof(int) * width, array_alloc.ptr(), 0,
                                   0, sizeof(int) * width, height, hipMemcpyDeviceToHost));
    ArrayFindIfNot(host_alloc.host_ptr(), 42, element_count);
  }
}
