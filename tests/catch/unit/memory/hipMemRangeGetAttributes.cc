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

TEST_CASE("Unit_hipMemRangeGetAttributes_Positive_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  LinearAllocGuard<void> allocation(LinearAllocs::hipMallocManaged, kPageSize);

  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetReadMostly, 0));
  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetPreferredLocation, 0));
  HIP_CHECK(hipMemPrefetchAsync(allocation.ptr(), kPageSize, hipCpuDeviceId));
  HIP_CHECK(hipMemAdvise(allocation.ptr(), kPageSize, hipMemAdviseSetAccessedBy, 0));

  constexpr size_t num_attributes = 4;
  std::array<hipMemRangeAttribute, num_attributes> attributes = {
      hipMemRangeAttributeReadMostly, hipMemRangeAttributePreferredLocation,
      hipMemRangeAttributeLastPrefetchLocation, hipMemRangeAttributeAccessedBy};

  std::array<int32_t*, num_attributes> data;
  for (auto& ptr : data) {
    ptr = new int32_t;
  }
  std::array<size_t, num_attributes> data_sizes = {4, 4, 4, 4};

  HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(data.data()), data_sizes.data(),
                                     attributes.data(), num_attributes, allocation.ptr(),
                                     kPageSize));

  REQUIRE(data[0][0] == 1);
  REQUIRE(data[1][0] == 0);
  REQUIRE(data[2][0] == hipCpuDeviceId);
  REQUIRE(data[3][0] == 0);

  for (auto ptr : data) {
    delete ptr;
  }
}

TEST_CASE("Unit_hipMemRangeGetAttributes_Negative_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory not supported");
    return;
  }

  constexpr size_t num_attributes = 4;
  hipMemRangeAttribute attributes[] = {
      hipMemRangeAttributeReadMostly, hipMemRangeAttributePreferredLocation,
      hipMemRangeAttributeLastPrefetchLocation, hipMemRangeAttributeAccessedBy};

  int32_t* data[num_attributes];
  for (auto& ptr : data) {
    ptr = new int32_t;
  }
  size_t data_sizes[] = {4, 4, 4, 4};

  LinearAllocGuard<void> managed(LinearAllocs::hipMallocManaged, kPageSize);

  SECTION("data == nullptr") {
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(nullptr, data_sizes, attributes, num_attributes,
                                             managed.ptr(), kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("data contains invalid pointers") {
    void* invalid_data[num_attributes] = {nullptr};
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(invalid_data, data_sizes, attributes, num_attributes,
                                             managed.ptr(), kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("data_sizes == nullptr") {
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(reinterpret_cast<void**>(data), nullptr, attributes,
                                             num_attributes, managed.ptr(), kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("data_sizes contains invalid values") {
    size_t invalid_data_sizes[] = {4, 5, 4, 6};
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(reinterpret_cast<void**>(data), invalid_data_sizes,
                                             attributes, num_attributes, managed.ptr(), kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("attributes == nullptr") {
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(reinterpret_cast<void**>(data), data_sizes, nullptr,
                                             num_attributes, managed.ptr(), kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("attributes contains invalid attributes") {
    hipMemRangeAttribute invalid_attributes[] = {
        hipMemRangeAttributeReadMostly, hipMemRangeAttributePreferredLocation,
        static_cast<hipMemRangeAttribute>(999), hipMemRangeAttributeAccessedBy};
    HIP_CHECK_ERROR(
        hipMemRangeGetAttributes(reinterpret_cast<void**>(data), data_sizes, invalid_attributes,
                                 num_attributes, managed.ptr(), kPageSize),
        hipErrorInvalidValue);
  }

  SECTION("num_attributes == 0") {
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(reinterpret_cast<void**>(data), data_sizes, attributes,
                                             0, managed.ptr(), kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("dev_ptr == nullptr") {
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(reinterpret_cast<void**>(data), data_sizes, attributes,
                                             num_attributes, nullptr, kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("dev_ptr is not managed memory") {
    LinearAllocGuard<void> non_managed(LinearAllocs::hipMalloc, kPageSize);
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(reinterpret_cast<void**>(data), data_sizes, attributes,
                                             num_attributes, non_managed.ptr(), kPageSize),
                    hipErrorInvalidValue);
  }

  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipMemRangeGetAttributes(reinterpret_cast<void**>(data), data_sizes, attributes,
                                             num_attributes, managed.ptr(), 0),
                    hipErrorInvalidValue);
  }

  for (auto ptr : data) {
    delete ptr;
  }
}