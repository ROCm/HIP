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

#include "memcpy2d_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE("Unit_hipMemcpyParam2DAsync_Positive_Basic") {
  using namespace std::placeholders;

  constexpr bool async = true;

  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  SECTION("Device to Host") {
    Memcpy2DDeviceToHostShell<async>(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, stream), stream);
  }
  SECTION("Device to Device") {
    SECTION("Peer access disabled") {
      Memcpy2DDeviceToDeviceShell<async, false>(
          std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, stream), stream);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDeviceToDeviceShell<async, true>(
          std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, stream), stream);
    }
  }
  SECTION("Host to Device") {
    Memcpy2DHostToDeviceShell<async>(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, stream), stream);
  }
  SECTION("Host to Host") {
    Memcpy2DHostToHostShell<async>(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, stream), stream);
  }
}

TEST_CASE("Unit_hipMemcpyParam2DAsync_Positive_Synchronization_Behavior") {
  using namespace std::placeholders;

  constexpr bool async = true;

  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") {
    Memcpy2DHtoDSyncBehavior(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, nullptr), false);
  }
  SECTION("Device to Pageable Host") {
    Memcpy2DDtoHPageableSyncBehavior(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, nullptr), true);
  }
  SECTION("Device to Pinned Host") {
    Memcpy2DDtoHPinnedSyncBehavior(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, nullptr), false);
  }
  SECTION("Device to Device") {
    Memcpy2DDtoDSyncBehavior(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, nullptr), false);
  }
  SECTION("Host to Host") {
    Memcpy2DHtoHSyncBehavior(
        std::bind(MemcpyParam2DAdapter<async>(), _1, _2, _3, _4, _5, _6, _7, nullptr), true);
  }
}

TEST_CASE("Unit_hipMemcpyParam2DAsync_Positive_Parameters") {
  constexpr bool async = true;
  Memcpy2DZeroWidthHeight<async>(MemcpyParam2DAdapter<async>());
}

TEST_CASE("Unit_hipMemcpyParam2DAsync_Positive_Array") {
  constexpr bool async = true;
  SECTION("Array from/to Host") {
    MemcpyParam2DArrayHostShell<async>(MemcpyParam2DAdapter<async>());
  }
  SECTION("Array from/to Device") {
    MemcpyParam2DArrayDeviceShell<async>(MemcpyParam2DAdapter<async>());
  }
}

TEST_CASE("Unit_hipMemcpyParam2DAsync_Negative_Parameters") {
  constexpr bool async = true;

  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  constexpr auto NegativeTests = [](void* dst, size_t dpitch, void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>()(static_cast<void*>(nullptr), dpitch, src,
                                                    spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>()(dst, dpitch, static_cast<void*>(nullptr),
                                                    spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("dstPitch < WidthInBytes") {
      HIP_CHECK_ERROR(
          MemcpyParam2DAdapter<async>()(dst, width - 1, src, spitch, width, height, kind),
          hipErrorInvalidValue);
    }
    SECTION("srcPitch < WidthInBytes") {
      HIP_CHECK_ERROR(
          MemcpyParam2DAdapter<async>()(dst, dpitch, src, width - 1, width, height, kind),
          hipErrorInvalidValue);
    }
    SECTION("dpitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>()(dst, static_cast<size_t>(attr) + 1, src, spitch,
                                                    width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("spitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>()(dst, dpitch, src, static_cast<size_t>(attr) + 1,
                                                    width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("WidthInBytes + srcXInBytes > srcPitch") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>(make_hipExtent(spitch - width + 1, 0, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("WidthInBytes + dstXInBytes > dstPitch") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>(make_hipExtent(0, 0, 0),
                                                  make_hipExtent(dpitch - width + 1, 0, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("srcY out of bounds") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>(make_hipExtent(0, 1, 0))(dst, dpitch, src, spitch,
                                                                           width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("dstY out of bounds") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>(make_hipExtent(0, 0, 0), make_hipExtent(0, 1, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("Invalid stream") {
      StreamGuard stream_guard(Streams::created);
      HIP_CHECK(hipStreamDestroy(stream_guard.stream()));
      HIP_CHECK_ERROR(MemcpyParam2DAdapter<async>()(dst, dpitch, src, spitch, width, height, kind,
                                                    stream_guard.stream()),
                      hipErrorContextIsDestroyed);
    }
  };

  SECTION("Host to device") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice);
  }

  SECTION("Device to host") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(host_alloc.ptr(), device_alloc.pitch(), device_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyDeviceToHost);
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    NegativeTests(dst_alloc.ptr(), cols * sizeof(int), src_alloc.ptr(), cols * sizeof(int),
                  cols * sizeof(int), rows, hipMemcpyHostToHost);
  }

  SECTION("Device to device") {
    LinearAllocGuard2D<int> src_alloc(cols, rows);
    LinearAllocGuard2D<int> dst_alloc(cols, rows);
    NegativeTests(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                  dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToDevice);
  }
}