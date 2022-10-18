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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

template <bool should_synchronize, typename F>
void Memcpy2DDeviceToHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyDeviceToHost, hipMemcpyDefault);

  constexpr size_t cols = 127;
  constexpr size_t rows = 128;

  LinearAllocGuard2D<int> device_alloc(cols, rows);

  const size_t host_pitch = GENERATE_REF(device_alloc.width(), device_alloc.width() + 64);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, host_pitch * rows);

  const dim3 threads_per_block(32, 32);
  const dim3 blocks(cols / threads_per_block.x + 1, rows / threads_per_block.y + 1);
  Iota<<<blocks, threads_per_block>>>(device_alloc.ptr(), device_alloc.pitch(),
                                      device_alloc.width_logical(), device_alloc.height(), 1);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(memcpy_func(host_alloc.ptr(), host_pitch, device_alloc.ptr(), device_alloc.pitch(),
                        device_alloc.width(), device_alloc.height(), kind));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  constexpr auto f = [](size_t x, size_t y, size_t z) { return z * cols * rows + y * cols + x; };
  PitchedMemoryVerify(host_alloc.ptr(), host_pitch, device_alloc.width_logical(),
                      device_alloc.height(), 1, f);
}

template <bool should_synchronize, bool enable_peer_access, typename F>
void Memcpy2DDeviceToDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyDeviceToDevice, hipMemcpyDefault);

  constexpr size_t cols = 127;
  constexpr size_t rows = 128;

  const auto device_count = HipTest::getDeviceCount();
  const auto src_device = GENERATE_COPY(range(0, device_count));
  const auto dst_device = GENERATE_COPY(range(0, device_count));
  const size_t src_cols_mult = GENERATE(1, 2);

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

  LinearAllocGuard2D<int> src_alloc(cols * src_cols_mult, rows);
  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocGuard2D<int> dst_alloc(cols, rows);
  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, dst_alloc.width() * rows);

  const dim3 threads_per_block(32, 32);
  const dim3 blocks(cols / threads_per_block.x + 1, rows / threads_per_block.y + 1);
  // Using dst_alloc width and height to set only the elements that will be copied over to
  // dst_alloc
  Iota<<<blocks, threads_per_block>>>(src_alloc.ptr(), src_alloc.pitch(), dst_alloc.width_logical(),
                                      dst_alloc.height(), 1);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(memcpy_func(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                        dst_alloc.width(), dst_alloc.height(), kind));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(hipMemcpy2D(host_alloc.ptr(), dst_alloc.width(), dst_alloc.ptr(), dst_alloc.pitch(),
                        dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToHost));
  constexpr auto f = [](size_t x, size_t y, size_t z) { return z * cols * rows + y * cols + x; };
  PitchedMemoryVerify(host_alloc.ptr(), dst_alloc.width(), dst_alloc.width_logical(),
                      dst_alloc.height(), 1, f);
}

template <bool should_synchronize, typename F>
void Memcpy2DHostToDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyHostToDevice, hipMemcpyDefault);

  constexpr size_t cols = 127;
  constexpr size_t rows = 128;

  LinearAllocGuard2D<int> device_alloc(cols, rows);

  const size_t host_pitch = GENERATE_REF(device_alloc.pitch(), 2 * device_alloc.pitch());

  LinearAllocGuard<int> src_host_alloc(LinearAllocs::hipHostMalloc, host_pitch * rows);
  LinearAllocGuard<int> dst_host_alloc(LinearAllocs::hipHostMalloc, device_alloc.width() * rows);

  constexpr auto f = [](size_t x, size_t y, size_t z) { return z * cols * rows + y * cols + x; };
  PitchedMemorySet(src_host_alloc.ptr(), host_pitch, device_alloc.width_logical(),
                   device_alloc.height(), 1, f);

  std::fill_n(dst_host_alloc.ptr(), device_alloc.width_logical() * rows, 0);

  HIP_CHECK(memcpy_func(device_alloc.ptr(), device_alloc.pitch(), src_host_alloc.ptr(), host_pitch,
                        device_alloc.width(), device_alloc.height(), kind));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(hipMemcpy2D(dst_host_alloc.ptr(), device_alloc.width(), device_alloc.ptr(),
                        device_alloc.pitch(), device_alloc.width(), device_alloc.height(),
                        hipMemcpyDeviceToHost));

  PitchedMemoryVerify(dst_host_alloc.ptr(), device_alloc.width(), device_alloc.width_logical(),
                      device_alloc.height(), 1, f);
}

template <bool should_synchronize, typename F>
void Memcpy2DHostToHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  const auto kind = GENERATE(hipMemcpyHostToHost, hipMemcpyDefault);

  constexpr size_t cols = 127;
  constexpr size_t rows = 128;

  const size_t src_pitch = GENERATE_REF(cols * sizeof(int), cols * sizeof(int) + 64);

  LinearAllocGuard<int> src_host(LinearAllocs::hipHostMalloc, src_pitch * rows);
  LinearAllocGuard<int> dst_host(LinearAllocs::hipHostMalloc, cols * sizeof(int) * rows);

  constexpr auto f = [](size_t x, size_t y, size_t z) { return z * cols * rows + y * cols + x; };
  PitchedMemorySet(src_host.ptr(), src_pitch, cols, rows, 1, f);

  HIP_CHECK(memcpy_func(dst_host.ptr(), cols * sizeof(int), src_host.ptr(), src_pitch,
                        cols * sizeof(int), rows, kind));
  if constexpr (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  PitchedMemoryVerify(dst_host.ptr(), cols * sizeof(int), cols, rows, 1, f);
}

// Synchronization behavior checks
template <typename F>
void MemcpySyncBehaviorCheck(F memcpy_func, const bool should_sync,
                             const hipStream_t kernel_stream) {
  LaunchDelayKernel(std::chrono::milliseconds{300}, kernel_stream);
  HIP_CHECK(memcpy_func());
  if (should_sync) {
    HIP_CHECK(hipStreamQuery(kernel_stream));
  } else {
    HIP_CHECK_ERROR(hipStreamQuery(kernel_stream), hipErrorNotReady);
  }
}

template <typename F>
void Memcpy2DHtoDSyncBehavior(F memcpy_func, const bool should_sync,
                              const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto host_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  LinearAllocGuard<int> host_alloc(host_alloc_type, 32 * sizeof(int) * 32);
  LinearAllocGuard2D<int> device_alloc(32, 32);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, device_alloc.ptr(), device_alloc.pitch(),
                                    host_alloc.ptr(), device_alloc.width(), device_alloc.width(),
                                    device_alloc.height(), hipMemcpyHostToDevice),
                          should_sync, kernel_stream);
}

template <typename F>
void Memcpy2DDtoHPageableSyncBehavior(F memcpy_func, const bool should_sync,
                                      const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, 32 * sizeof(int) * 32);
  LinearAllocGuard2D<int> device_alloc(32, 32);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), device_alloc.width(),
                                    device_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                    device_alloc.height(), hipMemcpyDeviceToHost),
                          should_sync, kernel_stream);
}

template <typename F>
void Memcpy2DDtoHPinnedSyncBehavior(F memcpy_func, const bool should_sync,
                                    const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, 32 * sizeof(int) * 32);
  LinearAllocGuard2D<int> device_alloc(32, 32);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), device_alloc.width(),
                                    device_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                    device_alloc.height(), hipMemcpyDeviceToHost),
                          should_sync, kernel_stream);
}

template <typename F>
void Memcpy2DDtoDSyncBehavior(F memcpy_func, const bool should_sync,
                              const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard2D<int> src_alloc(32, 32);
  LinearAllocGuard2D<int> dst_alloc(32, 32);
  MemcpySyncBehaviorCheck(
      std::bind(memcpy_func, dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToDevice),
      should_sync, kernel_stream);
}

template <typename F>
void Memcpy2DHtoHSyncBehavior(F memcpy_func, const bool should_sync,
                              const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto src_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto dst_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);

  LinearAllocGuard<int> src_alloc(src_alloc_type, 32 * sizeof(int) * 32);
  LinearAllocGuard<int> dst_alloc(dst_alloc_type, 32 * sizeof(int) * 32);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, dst_alloc.ptr(), 32 * sizeof(int), src_alloc.ptr(),
                                    32 * sizeof(int), 32 * sizeof(int), 32, hipMemcpyHostToHost),
                          should_sync, kernel_stream);
}

template <bool should_synchronize, typename F>
void Memcpy2DZeroWidthHeight(F memcpy_func, const hipStream_t stream = nullptr) {
  constexpr size_t cols = 63;
  constexpr size_t rows = 64;

  const auto [width_mult, height_mult] =
      GENERATE(std::make_pair(0, 1), std::make_pair(1, 0), std::make_pair(0, 0));

  SECTION("Device to Host") {
    LinearAllocGuard2D<uint8_t> device_alloc(cols, rows);
    LinearAllocGuard<uint8_t> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.width() * rows);
    std::fill_n(host_alloc.ptr(), device_alloc.width_logical() * device_alloc.height(), 42);
    HIP_CHECK(hipMemset2D(device_alloc.ptr(), device_alloc.pitch(), 1, device_alloc.width(),
                          device_alloc.height()));

    HIP_CHECK(memcpy_func(host_alloc.ptr(), device_alloc.width(), device_alloc.ptr(),
                          device_alloc.pitch(), device_alloc.width() * width_mult,
                          device_alloc.height() * height_mult, hipMemcpyDeviceToHost));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    ArrayFindIfNot(host_alloc.ptr(), static_cast<uint8_t>(42),
                   device_alloc.width_logical() * device_alloc.height());
  }

  SECTION("Device to Device") {
    LinearAllocGuard2D<uint8_t> src_alloc(cols, rows);
    LinearAllocGuard2D<uint8_t> dst_alloc(cols, rows);
    LinearAllocGuard<uint8_t> host_alloc(LinearAllocs::hipHostMalloc, dst_alloc.width() * rows);
    HIP_CHECK(
        hipMemset2D(src_alloc.ptr(), src_alloc.pitch(), 1, src_alloc.width(), src_alloc.height()));
    HIP_CHECK(
        hipMemset2D(dst_alloc.ptr(), dst_alloc.pitch(), 42, dst_alloc.width(), dst_alloc.height()));
    HIP_CHECK(memcpy_func(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                          dst_alloc.width() * width_mult, dst_alloc.height() * height_mult,
                          hipMemcpyDeviceToDevice));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipMemcpy2D(host_alloc.ptr(), dst_alloc.width(), dst_alloc.ptr(), dst_alloc.pitch(),
                          dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToHost));
    ArrayFindIfNot(host_alloc.ptr(), static_cast<uint8_t>(42),
                   dst_alloc.width_logical() * dst_alloc.height());
  }

  SECTION("Host to Device") {
    LinearAllocGuard2D<uint8_t> device_alloc(cols, rows);
    LinearAllocGuard<uint8_t> src_host_alloc(LinearAllocs::hipHostMalloc,
                                             device_alloc.width() * rows);
    LinearAllocGuard<uint8_t> dst_host_alloc(LinearAllocs::hipHostMalloc,
                                             device_alloc.width() * rows);
    std::fill_n(src_host_alloc.ptr(), device_alloc.width_logical() * device_alloc.height(), 1);
    HIP_CHECK(hipMemset2D(device_alloc.ptr(), device_alloc.pitch(), 42, device_alloc.width(),
                          device_alloc.height()));
    HIP_CHECK(memcpy_func(device_alloc.ptr(), device_alloc.pitch(), src_host_alloc.ptr(),
                          device_alloc.width(), device_alloc.width() * width_mult,
                          device_alloc.height() * height_mult, hipMemcpyHostToDevice));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipMemcpy2D(dst_host_alloc.ptr(), device_alloc.width(), device_alloc.ptr(),
                          device_alloc.pitch(), device_alloc.width(), device_alloc.height(),
                          hipMemcpyDeviceToHost));
    ArrayFindIfNot(dst_host_alloc.ptr(), static_cast<uint8_t>(42),
                   device_alloc.width_logical() * device_alloc.height());
  }

  SECTION("Host to Host") {
    const auto alloc_size = cols * rows;
    LinearAllocGuard<uint8_t> src_alloc(LinearAllocs::hipHostMalloc, alloc_size);
    LinearAllocGuard<uint8_t> dst_alloc(LinearAllocs::hipHostMalloc, alloc_size);
    std::fill_n(src_alloc.ptr(), alloc_size, 1);
    std::fill_n(dst_alloc.ptr(), alloc_size, 42);
    HIP_CHECK(memcpy_func(dst_alloc.ptr(), cols, src_alloc.ptr(), cols, cols * width_mult,
                          rows * height_mult, hipMemcpyHostToHost));
    if constexpr (should_synchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    ArrayFindIfNot(dst_alloc.ptr(), static_cast<uint8_t>(42), alloc_size);
  }
}