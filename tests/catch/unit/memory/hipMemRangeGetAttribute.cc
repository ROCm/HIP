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

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_ReadMostly_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeReadMostly, allocation.ptr(), kPageSize));
  
  REQUIRE(data == 0);

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetReadMostly, 0));
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeReadMostly, allocation.ptr(), kPageSize));
  
  REQUIRE(data == 1);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_ReadMostly_Partial_Range") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, 2 * kPageSize);

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetReadMostly, 0));

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeReadMostly, allocation.ptr(), 2 * kPageSize));

  REQUIRE(data == 0);

  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeReadMostly, allocation.ptr(), kPageSize));

  REQUIRE(data == 1);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_PreferredLocation_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributePreferredLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == hipInvalidDeviceId);

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetPreferredLocation, 0));
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributePreferredLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == 0);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_PreferredLocation_CPU") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetPreferredLocation, hipCpuDeviceId));

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributePreferredLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == hipCpuDeviceId);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_PreferredLocation_Partial_Range") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, 2 * kPageSize);

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetPreferredLocation, 0));

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributePreferredLocation, allocation.ptr(), 2 * kPageSize));

  REQUIRE(data == hipInvalidDeviceId);

  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributePreferredLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == 0);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_LastPrefetchLocation_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeLastPrefetchLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == hipInvalidDeviceId);

  HIP_CHECK(hipMemPrefetchAsync(allocation.ptr(), kPageSize, 0));
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeLastPrefetchLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == 0);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_LastPrefetchLocation_CPU") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  HIP_CHECK(hipMemPrefetchAsync(allocation.ptr(), kPageSize, hipCpuDeviceId));

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeLastPrefetchLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == hipCpuDeviceId);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_LastPrefetchLocation_Partial_Range") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, 2 * kPageSize);

  HIP_CHECK(hipMemPrefetchAsync(allocation.ptr(), kPageSize, 0));

  int32_t data;
  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeLastPrefetchLocation, allocation.ptr(), 2 * kPageSize));

  REQUIRE(data == hipInvalidDeviceId);

  HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(data), hipMemRangeAttributeLastPrefetchLocation, allocation.ptr(), kPageSize));

  REQUIRE(data == 0);
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_AccessedBy_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }
  
  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  std::array<int32_t, 4> data;
  HIP_CHECK(hipMemRangeGetAttribute(data.data(), sizeof(data), hipMemRangeAttributeAccessedBy, allocation.ptr(), kPageSize));

  for (auto device : data) {
    REQUIRE(device == hipInvalidDeviceId);
  }

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetAccessedBy, hipCpuDeviceId));
  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetAccessedBy, 0));
  HIP_CHECK(hipMemRangeGetAttribute(data.data(), sizeof(data), hipMemRangeAttributeAccessedBy, allocation.ptr(), kPageSize));

  // Use std::find since there is no guaranteed order in which devices will be returned
  REQUIRE(std::find(cbegin(data), cend(data), hipCpuDeviceId) != cend(data));
  REQUIRE(std::find(cbegin(data), cend(data), 0) != cend(data));
  
  // All the unused slots should be at the end
  for (auto it = cbegin(data) + 2; it != cend(data); ++it) {
    REQUIRE(*it == hipInvalidDeviceId);
  }
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_AccessedBy_Partial_Range") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }
  
  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, 2 * kPageSize);

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetAccessedBy, hipCpuDeviceId));
  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetAccessedBy, 0));

  std::array<int32_t, 4> data;
  HIP_CHECK(hipMemRangeGetAttribute(data.data(), sizeof(data), hipMemRangeAttributeAccessedBy, allocation.ptr(), 2 * kPageSize));

  for (auto device : data) {
    REQUIRE(device == hipInvalidDeviceId);
  }

  HIP_CHECK(hipMemRangeGetAttribute(data.data(), sizeof(data), hipMemRangeAttributeAccessedBy, allocation.ptr(), kPageSize));

  // Use std::find since there is no guaranteed order in which devices will be returned
  REQUIRE(std::find(cbegin(data), cend(data), hipCpuDeviceId) != cend(data));
  REQUIRE(std::find(cbegin(data), cend(data), 0) != cend(data));
  
  // All the unused slots should be at the end
  for (auto it = cbegin(data) + 2; it != cend(data); ++it) {
    REQUIRE(*it == hipInvalidDeviceId);
  }
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Positive_AccessedBy_MultiDevice") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Two or more device are required");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  std::vector<int32_t> data(device_count);
  HIP_CHECK(hipMemRangeGetAttribute(data.data(), sizeof(data), hipMemRangeAttributeAccessedBy, allocation.ptr(), kPageSize));

  for (auto device : data) {
    REQUIRE(device == hipInvalidDeviceId);
  }

  for (auto device = 0; device < device_count; ++device) {
    HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetAccessedBy, device));
  }

  HIP_CHECK(hipMemRangeGetAttribute(data.data(), sizeof(data), hipMemRangeAttributeAccessedBy, allocation.ptr(), kPageSize));

  // Use std::find since there is no guaranteed order in which devices will be returned
  for (auto device = 0; device < device_count; ++device) {
    REQUIRE(std::find(cbegin(data), cend(data), device) != cend(data));
  }
}

TEST_CASE("Unit_hipMemRangeGetAttribute_Negative_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }
  
  int32_t data;
  LinearAllocGuard<void> managed(LinearAllocs::hipMallocManaged, kPageSize);

  SECTION("data == nullptr") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(nullptr, 4, hipMemRangeAttributeReadMostly, managed.ptr(), kPageSize), hipErrorInvalidValue);
  }

  SECTION("data_size == 0") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 0, hipMemRangeAttributeReadMostly, managed.ptr(), kPageSize), hipErrorInvalidValue);
  }

  SECTION("data_size != 4 with hipMemRangeAttributeReadMostly") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 8, hipMemRangeAttributeReadMostly, managed.ptr(), kPageSize), hipErrorInvalidValue);
  }

  SECTION("data_size != 4 with hipMemRangeAttributePreferredLocation") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 8, hipMemRangeAttributePreferredLocation, managed.ptr(), kPageSize), hipErrorInvalidValue);
  }

  SECTION("data_size != 4 with hipMemRangeAttributeLastPrefetchLocation") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 8, hipMemRangeAttributeLastPrefetchLocation, managed.ptr(), kPageSize), hipErrorInvalidValue);
  }

  SECTION("data_size is not a multiple of 4 with hipMemRangeAttributeAccessedBy") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 10, hipMemRangeAttributeAccessedBy, managed.ptr(), kPageSize), hipErrorInvalidValue);
  }

  // TODO: Invalid attributes are accepted
  // SECTION("invalid attribute") {
  //   HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 4, static_cast<hipMemRangeAttribute>(999), managed.ptr(), kPageSize), hipErrorInvalidValue);
  // }

  SECTION("dev_ptr == nullptr") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 4, hipMemRangeAttributeReadMostly, nullptr, kPageSize), hipErrorInvalidValue);
  }

  SECTION("dev_ptr is not managed memory") {
    LinearAllocGuard<void> non_managed(LinearAllocs::hipMalloc, kPageSize);
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 4, hipMemRangeAttributeReadMostly, non_managed.ptr(), kPageSize), hipErrorInvalidValue);
  }

  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipMemRangeGetAttribute(&data, 4, hipMemRangeAttributeReadMostly, managed.ptr(), 0), hipErrorInvalidValue);
  }
}

// TODO: Write tests for coherency attribute on AMD