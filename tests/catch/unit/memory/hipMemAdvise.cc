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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

static inline hipMemoryAdvise GetUnsetMemAdvice(const hipMemoryAdvise advice) {
  switch (advice) {
    case hipMemAdviseSetAccessedBy:
      return hipMemAdviseUnsetAccessedBy;
    case hipMemAdviseSetReadMostly:
      return hipMemAdviseUnsetReadMostly;
    case hipMemAdviseSetPreferredLocation:
      return hipMemAdviseUnsetPreferredLocation;
    default:
      assert("Invalid hipMemoryAdvise enumerator");
  }
}

static inline hipMemRangeAttribute GetMemAdviceAttr(const hipMemoryAdvise advice) {
  switch (advice) {
    case hipMemAdviseSetAccessedBy:
      return hipMemRangeAttributeAccessedBy;
    case hipMemAdviseSetReadMostly:
      return hipMemRangeAttributeReadMostly;
    case hipMemAdviseSetPreferredLocation:
      return hipMemRangeAttributePreferredLocation;
    default:
      assert("Invalid hipMemoryAdvise enumerator");
  }
}

std::vector<int> GetDevicesWithAdviseSupport() {
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

TEST_CASE("Unit_hipMemAdvise_Set_Unset_Basic") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
    return;
  }
  supported_devices.push_back(hipCpuDeviceId);
  const auto device = GENERATE_COPY(from_range(supported_devices));

  const auto SetUnset = [=](const hipMemoryAdvise advice) {
    LinearAllocGuard<uint8_t> alloc(LinearAllocs::hipMallocManaged, kPageSize);
    int32_t attribute = 0u;
    HIP_CHECK(hipMemAdvise(alloc.ptr(), kPageSize, advice, device));
    HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute), GetMemAdviceAttr(advice),
                                      alloc.ptr(), kPageSize));
    REQUIRE((advice == hipMemAdviseSetReadMostly ? 1 : device) == attribute);
    HIP_CHECK(hipMemAdvise(alloc.ptr(), kPageSize, GetUnsetMemAdvice(advice), device));
    HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute), GetMemAdviceAttr(advice),
                                      alloc.ptr(), kPageSize));
    REQUIRE((advice == hipMemAdviseSetReadMostly ? 0 : hipInvalidDeviceId) == attribute);
  };

  SECTION("hipMemAdviseSetAccessedBy") { SetUnset(hipMemAdviseSetAccessedBy); }
  SECTION("hipMemAdviseSetReadMostly") { SetUnset(hipMemAdviseSetReadMostly); }
  SECTION("hipMemAdviseSetPreferredLocation") { SetUnset(hipMemAdviseSetPreferredLocation); }
}

TEST_CASE("Unit_hipMemAdvise_No_Flag_Interference") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
    return;
  }
  supported_devices.push_back(hipCpuDeviceId);
  const auto device = GENERATE_COPY(from_range(supported_devices));

  std::array<hipMemoryAdvise, 3> advice{hipMemAdviseSetReadMostly, hipMemAdviseSetPreferredLocation,
                                        hipMemAdviseSetAccessedBy};
  for (int i = 0; i < 6; ++i) {
    std::next_permutation(std::begin(advice), std::end(advice));
    LinearAllocGuard<void> alloc(LinearAllocs::hipMallocManaged, kPageSize);

    for (const auto a : advice) {
      HIP_CHECK(hipMemAdvise(alloc.ptr(), kPageSize, a, device));
    }

    for (const auto a : advice) {
      auto attribute = 0u;
      HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute), GetMemAdviceAttr(a),
                                        alloc.ptr(), kPageSize));
      REQUIRE((a == hipMemAdviseSetReadMostly ? 1 : device) == attribute);
    }
  }
}

TEST_CASE("Unit_hipMemAdvise_Rounding") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
    return;
  }
  supported_devices.push_back(hipCpuDeviceId);
  const auto device = supported_devices.front();

  LinearAllocGuard<uint8_t> alloc(LinearAllocs::hipMallocManaged, 3 * kPageSize);
  REQUIRE_FALSE(reinterpret_cast<intptr_t>(alloc.ptr()) % kPageSize);
  const auto [offset, width] =
      GENERATE_COPY(std::make_pair(kPageSize / 4, kPageSize / 2),   // Withing page
                    std::make_pair(kPageSize / 2, kPageSize),       // Across page border
                    std::make_pair(kPageSize / 2, kPageSize * 2));  // Across two page borders
  HIP_CHECK(hipMemAdvise(alloc.ptr() + offset, width, hipMemAdviseSetAccessedBy, device));
  constexpr auto RoundDown = [](const intptr_t a, const intptr_t n) { return a - a % n; };
  constexpr auto RoundUp = [RoundDown](const intptr_t a, const intptr_t n) {
    return RoundDown(a + n - 1, n);
  };
  const auto base = alloc.ptr();
  const auto rounded_up = RoundUp(offset + width, kPageSize);
  unsigned int attribute = 0;
  HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute), hipMemRangeAttributeAccessedBy,
                                    reinterpret_cast<void*>(base), rounded_up));
  REQUIRE(device == attribute);
  HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute), hipMemRangeAttributeAccessedBy,
                                    alloc.ptr(), 3 * kPageSize));
  REQUIRE((rounded_up == 3 * kPageSize ? device : hipInvalidDeviceId) == attribute);
}

TEST_CASE("Unit_hipMemAdvise_Flags_Do_Not_Cause_Prefetch") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
  }
  supported_devices.push_back(hipCpuDeviceId);

  const auto Test = [](const int device, const hipMemoryAdvise advice) {
    LinearAllocGuard<void> alloc(LinearAllocs::hipMallocManaged, kPageSize);
    HIP_CHECK(hipMemAdvise(alloc.ptr(), kPageSize, hipMemAdviseSetPreferredLocation, device));
    int32_t attribute = 0u;
    HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute),
                                      hipMemRangeAttributeLastPrefetchLocation, alloc.ptr(),
                                      kPageSize));
    REQUIRE(attribute == hipInvalidDeviceId);
  };
  const auto device =
      GENERATE_COPY(from_range(std::begin(supported_devices), std::end(supported_devices)));

  SECTION("hipMemAdviseSetPreferredLocation") { Test(device, hipMemAdviseSetPreferredLocation); }
  SECTION("hipMemAdviseSetAccessedBy") { Test(device, hipMemAdviseSetAccessedBy); }
}

TEST_CASE("Unit_hipMemAdvise_Read_Write_After_Advise") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
  }
  LinearAllocGuard<int> alloc(LinearAllocs::hipMallocManaged, kPageSize);
  constexpr size_t count = kPageSize / sizeof(*alloc.ptr());

  const auto ReadWriteManagedMemory = [&](const int device, const hipMemoryAdvise advice) {
    HIP_CHECK(hipMemAdvise(alloc.ptr(), kPageSize, advice, device));

    std::fill_n(alloc.ptr(), count, -1);
    ArrayFindIfNot(alloc.ptr(), -1, count);
    for (int i = 0; i < supported_devices.size(); ++i) {
      HIP_CHECK(hipSetDevice(supported_devices[i]));
      VectorIncrement<<<count / 1024 + 1, 1024>>>(alloc.ptr(), 1, count);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      ArrayFindIfNot(alloc.ptr(), i, count);
    }

    int32_t attribute = 0u;
    HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute), GetMemAdviceAttr(advice),
                                      alloc.ptr(), kPageSize));
    REQUIRE((advice == hipMemAdviseSetReadMostly ? 1 : device) == attribute);
  };

  SECTION("ReadMostly") { ReadWriteManagedMemory(hipInvalidDeviceId, hipMemAdviseSetReadMostly); }
  supported_devices.push_back(hipCpuDeviceId);
  const auto device =
      GENERATE_COPY(from_range(std::begin(supported_devices), std::end(supported_devices)));
  supported_devices.pop_back();
  SECTION("PreferredLocation") { ReadWriteManagedMemory(device, hipMemAdviseSetPreferredLocation); }
  SECTION("AccessedBy") { ReadWriteManagedMemory(device, hipMemAdviseSetAccessedBy); }
}

TEST_CASE("Unit_hipMemAdvise_Prefetch_After_Advise") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
  }
  supported_devices.push_back(hipCpuDeviceId);
  const auto advice = GENERATE(hipMemAdviseSetAccessedBy, hipMemAdviseSetReadMostly,
                               hipMemAdviseSetPreferredLocation);
  const auto device = GENERATE_COPY(from_range(supported_devices));

  LinearAllocGuard<int> alloc(LinearAllocs::hipMallocManaged, kPageSize);
  HIP_CHECK(hipMemAdvise(alloc.ptr(), kPageSize, advice, device));

  for (const auto d : supported_devices) {
    HIP_CHECK(hipMemPrefetchAsync(alloc.ptr(), kPageSize, d));
    HIP_CHECK(hipStreamSynchronize(nullptr));
    int32_t attribute = 0;
    HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute),
                                      hipMemRangeAttributeLastPrefetchLocation, alloc.ptr(),
                                      kPageSize));
    REQUIRE(d == attribute);
  }

  int32_t attribute = 0;
  HIP_CHECK(hipMemRangeGetAttribute(&attribute, sizeof(attribute), GetMemAdviceAttr(advice),
                                    alloc.ptr(), kPageSize));
  REQUIRE((advice == hipMemAdviseSetReadMostly ? 1 : device) == attribute);
}

TEST_CASE("Unit_hipMemAdvise_AccessedBy_All_Devices") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
    return;
  }
  supported_devices.push_back(hipCpuDeviceId);

  LinearAllocGuard<void> alloc(LinearAllocs::hipMallocManaged, kPageSize);
  for (const auto device : supported_devices) {
    HIP_CHECK(hipMemAdvise(alloc.ptr(), kPageSize, hipMemAdviseSetAccessedBy, device));
  }
  std::vector<int> accessed_by(supported_devices.size(), hipInvalidDeviceId);
  HIP_CHECK(hipMemRangeGetAttribute(accessed_by.data(), sizeof(accessed_by.data()),
                                    hipMemRangeAttributeAccessedBy, alloc.ptr(), kPageSize));
  REQUIRE_THAT(accessed_by, Catch::Matchers::Equals(supported_devices));
}

TEST_CASE("Unit_hipMemAdvise_Negative_Parameters") {
  auto supported_devices = GetDevicesWithAdviseSupport();
  if (supported_devices.empty()) {
    HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory");
  }
  const auto device = supported_devices.front();

  LinearAllocGuard<void> alloc(LinearAllocs::hipMallocManaged, kPageSize);

  SECTION("Invalid advice") {
    HIP_CHECK_ERROR(hipMemAdvise(alloc.ptr(), kPageSize, static_cast<hipMemoryAdvise>(-1), device),
                    hipErrorInvalidValue);
  }
  const auto advice = GENERATE(hipMemAdviseSetAccessedBy, hipMemAdviseSetReadMostly,
                               hipMemAdviseSetPreferredLocation);
  SECTION("dev_ptr == nullptr") {
    HIP_CHECK_ERROR(hipMemAdvise(nullptr, kPageSize, advice, device), hipErrorInvalidValue);
  }
  SECTION("dev_ptr pointing to non-managed memory") {
    LinearAllocGuard<void> alloc(LinearAllocs::hipMalloc, kPageSize);
    HIP_CHECK_ERROR(hipMemAdvise(alloc.ptr(), kPageSize, advice, device), hipErrorInvalidValue);
  }
  SECTION("Invalid device") {
    HIP_CHECK_ERROR(hipMemAdvise(alloc.ptr(), kPageSize, advice, hipInvalidDeviceId),
                    (advice == hipMemAdviseSetReadMostly ? hipSuccess : hipErrorInvalidDevice));
  }
}