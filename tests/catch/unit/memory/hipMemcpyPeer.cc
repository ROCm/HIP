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
/*
Testcase Scenarios :
Unit_hipMemcpyPeer_Positive_Default - Test basic P2P memcpy between two devices with hipMemcpyPeer api
Unit_hipMemcpyPeer_Positive_Synchronization_Behavior - Test synchronization behavior for hipMemcpyPeer api
Unit_hipMemcpyPeer_Positive_ZeroSize - Test that no data is copied when sizeBytes is set to 0
Unit_hipMemcpyPeer_Negative_Parameters - Test unsuccessful execution of hipMemcpyPeer api when parameters are invalid
*/
#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

TEST_CASE("Unit_hipMemcpyPeer_Positive_Default") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);

  int can_access_peer = 0;
  const auto src_device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto dst_device = GENERATE(range(0, HipTest::getDeviceCount()));
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, allocation_size);
    LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, allocation_size);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, allocation_size);

    const auto element_count = allocation_size / sizeof(*src_alloc.ptr());
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    constexpr int expected_value = 22;
    HIP_CHECK(hipSetDevice(src_device));
    VectorSet<<<block_count, thread_count, 0>>>(src_alloc.ptr(), expected_value,
                                                             element_count);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, allocation_size));

    HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_alloc.ptr(), allocation_size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));

    ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

TEST_CASE("Unit_hipMemcpyPeer_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());

  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int can_access_peer = 0;
  const auto src_device = 0;
  const auto dst_device = 1;

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    HIP_CHECK(hipSetDevice(src_device));
    LaunchDelayKernel(std::chrono::milliseconds{100}, nullptr);

    HIP_CHECK(hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, kPageSize));
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

TEST_CASE("Unit_hipMemcpyPeer_Positive_ZeroSize") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  const auto allocation_size = kPageSize;

  int can_access_peer = 0;
  const auto src_device = 0;
  const auto dst_device = 1;

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, allocation_size);
    LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, allocation_size, hipHostMallocPortable);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, allocation_size);

    const auto element_count = allocation_size / sizeof(*src_alloc.ptr());
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    constexpr int set_value = 22;
    HIP_CHECK(hipSetDevice(src_device));
    VectorSet<<<block_count, thread_count, 0>>>(src_alloc.ptr(), set_value,
                                                             element_count);
    HIP_CHECK(hipGetLastError());

    constexpr int expected_value = 21;
    std::fill_n(src_alloc.host_ptr(), element_count, expected_value);

    HIP_CHECK(hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, 0));

    HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_alloc.ptr(), allocation_size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));

    ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}

TEST_CASE("Unit_hipMemcpyPeer_Negative_Parameters") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int can_access_peer = 0;
  const auto src_device = 0;
  const auto dst_device = 1;

  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));

    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    HIP_CHECK(hipSetDevice(src_device));

    SECTION("Nullptr to Destination Pointer") {
      HIP_CHECK_ERROR(hipMemcpyPeer(nullptr, dst_device, src_alloc.ptr(), src_device, kPageSize), hipErrorInvalidValue);
    }

    SECTION("Nullptr to Source Pointer") {
      HIP_CHECK_ERROR(hipMemcpyPeer(dst_alloc.ptr(), dst_device, nullptr, src_device, kPageSize), hipErrorInvalidValue);
    }

    SECTION("Passing more than allocated size") {
      HIP_CHECK_ERROR(hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), src_device, kPageSize + 1), hipErrorInvalidValue);
    }

    SECTION("Passing invalid Destination device ID") {
      HIP_CHECK_ERROR(hipMemcpyPeer(dst_alloc.ptr(), device_count, src_alloc.ptr(), src_device, kPageSize), hipErrorInvalidDevice);
    }

    SECTION("Passing invalid Source device ID") {
      HIP_CHECK_ERROR(hipMemcpyPeer(dst_alloc.ptr(), dst_device, src_alloc.ptr(), device_count, kPageSize), hipErrorInvalidDevice);
    }

    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  } else {
    INFO("Peer access cannot be enabled between devices " << src_device << " " << dst_device);
  }
}
