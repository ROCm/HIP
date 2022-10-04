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

#include <vector>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

std::vector<int> GetDevicesWithPrefetchSupport() {
  const auto device_count = HipTest::getDeviceCount();
  std::vector<int> supported_devices;
  supported_devices.reserve(device_count + 1);
  for (int i = 0; i < device_count; ++i) {
    if (DeviceAttributesSupport(i, hipDeviceAttributeManagedMemory,
                                hipDeviceAttributeConcurrentManagedAccess)) {
      supported_devices.push_back(i);
    }
  }
  return supported_devices;
}

__global__ void MemPrefetchAsyncKernel(int* C_d, const int* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

TEST_CASE("Unit_hipMemPrefetchAsync_Basic") {
  const auto supported_devices = GetDevicesWithPrefetchSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test need at least one device with managed memory support");
  }

  LinearAllocGuard<int> alloc1(LinearAllocs::hipMallocManaged, kPageSize);
  const auto count = kPageSize / sizeof(*alloc1.ptr());
  constexpr auto fill_value = 42;
  std::fill_n(alloc1.ptr(), count, fill_value);

  for (const auto device : supported_devices) {
    HIP_CHECK(hipSetDevice(device));
    LinearAllocGuard<int> alloc2(LinearAllocs::hipMallocManaged, kPageSize);
    StreamGuard sg(Streams::created);
    HIP_CHECK(hipMemPrefetchAsync(alloc1.ptr(), kPageSize, device, sg.stream()));
    MemPrefetchAsyncKernel<<<count / 1024 + 1, 1024, 0, sg.stream()>>>(alloc2.ptr(), alloc1.ptr(),
                                                                       count);
    HIP_CHECK(hipStreamSynchronize(sg.stream()));
    ArrayFindIfNot(alloc1.ptr(), fill_value, count);
    ArrayFindIfNot(alloc2.ptr(), fill_value * fill_value, count);
  }

  HIP_CHECK(hipMemPrefetchAsync(alloc1.ptr(), kPageSize, hipCpuDeviceId));
  HIP_CHECK(hipStreamSynchronize(nullptr));
  ArrayFindIfNot(alloc1.ptr(), fill_value, count);
}

TEST_CASE("Unit_hipMemPrefetchAsync_Negative_Parameters") {
  auto supported_devices = GetDevicesWithPrefetchSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test need at least one device with managed memory support");
  }
  supported_devices.push_back(hipCpuDeviceId);
  const auto device = GENERATE_COPY(from_range(supported_devices));

  LinearAllocGuard<void> alloc(LinearAllocs::hipMallocManaged, kPageSize);
  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipMemPrefetchAsync(alloc.ptr(), 0, device), hipErrorInvalidValue);
  }
  SECTION("count larger than allocation size") {
    HIP_CHECK_ERROR(hipMemPrefetchAsync(alloc.ptr(), kPageSize + 1, device), hipErrorInvalidValue);
  }
  SECTION("Invalid device") {
    HIP_CHECK_ERROR(hipMemPrefetchAsync(alloc.ptr(), kPageSize, hipInvalidDeviceId),
                    hipErrorInvalidDevice);
  }
  SECTION("Invalid stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipMemPrefetchAsync(alloc.ptr(), kPageSize, device, stream),
                    hipErrorContextIsDestroyed);
  }
}