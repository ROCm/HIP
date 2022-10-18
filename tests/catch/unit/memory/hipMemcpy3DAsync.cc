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

#include "memcpy3d_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Basic") {
  constexpr bool async = true;

  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  SECTION("Device to Host") { Memcpy3DDeviceToHostShell<async>(Memcpy3DWrapper<async>, stream); }

  SECTION("Device to Device") {
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(Memcpy3DWrapper<async>, stream);
    }
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(Memcpy3DWrapper<async>, stream);
    }
  }

  SECTION("Host to Device") { Memcpy3DHostToDeviceShell<async>(Memcpy3DWrapper<async>, stream); }

  SECTION("Host to Host") { Memcpy3DHostToHostShell<async>(Memcpy3DWrapper<async>, stream); }
}

TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Synchronization_Behavior") {
  constexpr bool async = true;

  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy3DHtoDSyncBehavior(Memcpy3DWrapper<async>, false); }

  SECTION("Device to Pageable Host") {
    Memcpy3DDtoHPageableSyncBehavior(Memcpy3DWrapper<async>, true);
  }

  SECTION("Device to Pinned Host") {
    Memcpy3DDtoHPinnedSyncBehavior(Memcpy3DWrapper<async>, false);
  }

  SECTION("Device to Device") { Memcpy3DDtoDSyncBehavior(Memcpy3DWrapper<async>, false); }

  SECTION("Host to Host") { Memcpy3DHtoHSyncBehavior(Memcpy3DWrapper<async>, true); }
}

TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Parameters") {
  constexpr bool async = true;
  Memcpy3DZeroWidthHeightDepth<async>(Memcpy3DWrapper<async>);
}

TEST_CASE("Unit_hipMemcpy3DAsync_Positive_Array") {
  constexpr bool async = true;
  SECTION("Array from/to Host") { Memcpy3DArrayHostShell<async>(Memcpy3DWrapper<async>); }
  SECTION("Array from/to Device") { Memcpy3DArrayDeviceShell<async>(Memcpy3DWrapper<async>); }
}

TEST_CASE("Unit_hipMemcpy3DAsync_Negative_Parameters") {
  constexpr bool async = true;
  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  constexpr auto NegativeTests = [](hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                    hipPos src_pos, hipExtent extent, hipMemcpyKind kind) {
    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("src_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("dst_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("extent.width + dst_pos.x > dst_ptr.pitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("extent.width + src_pos.x > src_ptr.pitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.y out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.y out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dst_pos.z out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src_pos.z out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, src_pos, extent,
                                             static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      StreamGuard stream_guard(Streams::created);
      HIP_CHECK(hipStreamDestroy(stream_guard.stream()));
      HIP_CHECK_ERROR(Memcpy3DWrapper<async>(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind,
                                             stream_guard.stream()),
                      hipErrorContextIsDestroyed);
    }
  };

  SECTION("Host to Device") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice);
  }

  SECTION("Device to Host") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0), extent,
                  hipMemcpyDeviceToHost);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    NegativeTests(make_hipPitchedPtr(dst_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(src_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToHost);
  }

  SECTION("Device to Device") {
    LinearAllocGuard3D<int> src_alloc(extent);
    LinearAllocGuard3D<int> dst_alloc(extent);
    NegativeTests(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                  make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice);
  }
}