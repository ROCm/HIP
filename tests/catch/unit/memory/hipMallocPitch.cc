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
#include <initializer_list>
#include <memory>
#include "hip/driver_types.h"
#include <cstring>
#include <vector>
#include <limits>

/**
 * @brief Test hipMalloc3D, hipMallocPitch and hipMemAllocPitch with multiple input values.
 *        Checks that the memory has been allocated with the specified pitch and extent sizes.
 */

struct MemoryInfo {
  size_t freeMem;
  size_t totalMem;
};

inline static MemoryInfo createMemoryInfo() {
  MemoryInfo memoryInfo{};
  HIP_CHECK(hipMemGetInfo(&memoryInfo.freeMem, &memoryInfo.totalMem));
  return memoryInfo;
}

static void validateMemory(void* devPtr, hipExtent extent, size_t pitch,
                           MemoryInfo memBeforeAllocation) {
  INFO("Width: " << extent.width << " Height: " << extent.height << " Depth: " << extent.depth);

  MemoryInfo memAfterAllocation{createMemoryInfo()};
  const size_t theoreticalAllocatedMemory{pitch * extent.height * extent.depth};
  const size_t allocatedMemory = memBeforeAllocation.freeMem - memAfterAllocation.freeMem;

  if (theoreticalAllocatedMemory == 0) {
    REQUIRE(theoreticalAllocatedMemory == allocatedMemory);
    return; /* If there was no memory allocated then we don't need to do further checks. */
  } else {
    REQUIRE(theoreticalAllocatedMemory <= allocatedMemory);
  }

  // void* hostPtr = std::malloc(theoreticalAllocatedMemory);
  std::unique_ptr<char[]> hostPtr{new char[theoreticalAllocatedMemory]};
  std::memset(hostPtr.get(), 2, theoreticalAllocatedMemory);

  hipPitchedPtr devPitchedPtr{devPtr, pitch, extent.width, extent.height};
  hipPitchedPtr hostPitchedPtr{hostPtr.get(), pitch, extent.width, extent.height};

  HIP_CHECK(hipMemset3D(devPitchedPtr, 1, extent));

  hipMemcpy3DParms params{};
  params.srcPtr = devPitchedPtr;
  params.kind = hipMemcpyKind::hipMemcpyDeviceToHost;
  params.dstPtr = hostPitchedPtr;
  params.extent = extent;
  HIP_CHECK(hipMemcpy3D(&params))

  bool mismatch = false;
  for (size_t width = 0; width < extent.width; ++width) {
    for (size_t height = 0; height < extent.height; ++height) {
      for (size_t depth = 0; depth < extent.depth; ++depth) {
        char* reinterpretedPtr = reinterpret_cast<char*>(hostPtr.get());
        size_t index = (pitch * extent.height * depth) + (pitch * height) + width;
        if (*(reinterpretedPtr + index) != 1) {
          mismatch = true;
        }
      }
    }
  }
  REQUIRE(!mismatch);
}

class ExtentGenerator {
 public:
  static constexpr size_t totalRandomValues{20};
  static constexpr size_t seed{1337};

  std::uniform_int_distribution<size_t> width_distribution{1, 1024};
  std::uniform_int_distribution<size_t> height_distribution{1, 100};
  std::uniform_int_distribution<size_t> depth_distribution{1, 100};

  std::vector<hipExtent> extents2D{};
  std::vector<hipExtent> extents3D{};

  static ExtentGenerator& getInstance() {
    static ExtentGenerator instance;
    return instance;
  }

 private:
  ExtentGenerator() {
    std::mt19937 randomGenerator{seed};
    extents3D = std::vector<hipExtent>{hipExtent{0, 0, 0}, hipExtent{1, 0, 0}, hipExtent{0, 1, 0},
                                       hipExtent{0, 0, 1}};

    for (size_t i = 0; i < totalRandomValues; ++i) {
      extents3D.push_back(hipExtent{width_distribution(randomGenerator),
                                    height_distribution(randomGenerator),
                                    depth_distribution(randomGenerator)});
    }

    extents2D = std::vector<hipExtent>{hipExtent{0, 0, 1}, hipExtent{1, 0, 1}, hipExtent{0, 1, 1}};

    for (size_t i = 0; i < totalRandomValues; ++i) {
      extents2D.push_back(
          hipExtent{width_distribution(randomGenerator), height_distribution(randomGenerator), 1});
    }
  }
};

enum class AllocationApi { hipMalloc3D, hipMallocPitch, hipMemAllocPitch };

hipExtent generateExtent(AllocationApi api) {
  hipExtent extent;
  if (api == AllocationApi::hipMalloc3D) {
    auto& extents3D = ExtentGenerator::getInstance().extents3D;
    extent = GENERATE_REF(from_range(extents3D.begin(), extents3D.end()));
  } else {
    auto& extents2D = ExtentGenerator::getInstance().extents2D;
    extent = GENERATE_REF(from_range(extents2D.begin(), extents2D.end()));
  }

  return extent;
}


TEST_CASE("Unit_hipMalloc3D_ValidatePitch") {
  hipPitchedPtr hipPitchedPtr;
  hipExtent validExtent{generateExtent(AllocationApi::hipMalloc3D)};

  MemoryInfo memBeforeAllocation{createMemoryInfo()};
  HIP_CHECK(hipMalloc3D(&hipPitchedPtr, validExtent));
  validateMemory(hipPitchedPtr.ptr, validExtent, hipPitchedPtr.pitch, memBeforeAllocation);
  HIP_CHECK(hipFree(hipPitchedPtr.ptr));
}

TEST_CASE("Unit_hipMemAllocPitch_ValidatePitch") {
  size_t pitch;
  hipDeviceptr_t ptr;
  hipExtent validExtent{generateExtent(AllocationApi::hipMemAllocPitch)};
  MemoryInfo memBeforeAllocation{createMemoryInfo()};
  unsigned int elementSizeBytes = GENERATE(4, 8, 16);

#if HT_NVIDIA /* EXSWCPHIPT-46 */
  if (validExtent.width == 0 || validExtent.height == 0) {
    return;
  }
#endif

  HIP_CHECK(
      hipMemAllocPitch(&ptr, &pitch, validExtent.width, validExtent.height, elementSizeBytes));
  validateMemory(reinterpret_cast<void*>(ptr), validExtent, pitch, memBeforeAllocation);
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
}

TEST_CASE("Unit_hipMallocPitch_ValidatePitch") {
  size_t pitch;
  void* ptr;
  hipExtent validExtent{generateExtent(AllocationApi::hipMemAllocPitch)};
  MemoryInfo memBeforeAllocation{createMemoryInfo()};
  HIP_CHECK(hipMallocPitch(&ptr, &pitch, validExtent.width, validExtent.height));
  validateMemory(ptr, validExtent, pitch, memBeforeAllocation);
  HIP_CHECK(hipFree(ptr));
}

TEST_CASE("Unit_hipMalloc3D_Negative") {
  SECTION("Invalid ptr") {
    hipExtent validExtent{1, 1, 1};
    HIP_CHECK_ERROR(hipMalloc3D(nullptr, validExtent), hipErrorInvalidValue);
  }

  hipPitchedPtr ptr;
  constexpr size_t maxSizeT = std::numeric_limits<size_t>::max();

  SECTION("Max size_t width") {
    hipExtent validExtent{maxSizeT, 1, 1};
#if HT_AMD /* EXSWCPHIPT-46 */
    HIP_CHECK_ERROR(hipMalloc3D(&ptr, validExtent), hipErrorOutOfMemory);
#else
    HIP_CHECK_ERROR(hipMalloc3D(&ptr, validExtent), hipErrorInvalidValue);
#endif
  }

  SECTION("Max size_t height") {
    hipExtent validExtent{1, maxSizeT, 1};
    HIP_CHECK_ERROR(hipMalloc3D(&ptr, validExtent), hipErrorOutOfMemory);
  }

  SECTION("Max size_t width") {
    hipExtent validExtent{1, 1, maxSizeT};
    HIP_CHECK_ERROR(hipMalloc3D(&ptr, validExtent), hipErrorOutOfMemory);
  }
}

TEST_CASE("Unit_hipMallocPitch_Negative") {
  size_t pitch;
  void* ptr;
  constexpr size_t maxSizeT = std::numeric_limits<size_t>::max();

  SECTION("Invalid ptr") {
    HIP_CHECK_ERROR(hipMallocPitch(nullptr, &pitch, 1, 1), hipErrorInvalidValue);
  }

#if !HT_AMD /* EXSWCPHIPT-48 */
  SECTION("Invalid pitch") {
    HIP_CHECK_ERROR(hipMallocPitch(&ptr, nullptr, 1, 1), hipErrorInvalidValue);
  }
#endif

  SECTION("Max size_t width") {
#if HT_AMD /* EXSWCPHIPT-46 */
    HIP_CHECK_ERROR(hipMallocPitch(&ptr, &pitch, maxSizeT, 1), hipErrorOutOfMemory);
#else
    HIP_CHECK_ERROR(hipMallocPitch(&ptr, &pitch, maxSizeT, 1), hipErrorInvalidValue);
#endif
  }

  SECTION("Max size_t height") {
    HIP_CHECK_ERROR(hipMallocPitch(&ptr, &pitch, 1, maxSizeT), hipErrorOutOfMemory);
  }
}

TEST_CASE("Unit_hipMemAllocPitch_Negative") {
  size_t pitch;
  hipDeviceptr_t ptr{};
  unsigned int validElementSizeBytes{4};
  constexpr size_t maxSizeT = std::numeric_limits<size_t>::max();

#if HT_NVIDIA
  /* Device synchronize is used here to initialize the device.
   * Nvidia does not implicitly do it for this Api. And hipInit(0) does not work either.
   */
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Invalid elementSizeBytes") {
    unsigned int invalidElementSizeBytes = GENERATE(0, 7, 12, 17);
    HIP_CHECK_ERROR(hipMemAllocPitch(&ptr, &pitch, 1, 1, invalidElementSizeBytes),
                    hipErrorInvalidValue);
  }

  SECTION("Zero width") {
    HIP_CHECK_ERROR(hipMemAllocPitch(&ptr, &pitch, 0, 1, validElementSizeBytes),
                    hipErrorInvalidValue);
  }
  SECTION("Zero height") {
    HIP_CHECK_ERROR(hipMemAllocPitch(&ptr, &pitch, 1, 0, validElementSizeBytes),
                    hipErrorInvalidValue);
  }
#endif

  SECTION("Invalid dptr") {
    HIP_CHECK_ERROR(hipMemAllocPitch(nullptr, &pitch, 1, 1, validElementSizeBytes),
                    hipErrorInvalidValue);
  }

#if !HT_AMD /* EXSWCPHIPT-48 */
  SECTION("Invalid pitch") {
    HIP_CHECK_ERROR(hipMemAllocPitch(&ptr, nullptr, 1, 1, validElementSizeBytes),
                    hipErrorInvalidValue);
  }
#endif

  SECTION("Max size_t width") {
#if HT_AMD /* EXSWCPHIPT-46 */
    HIP_CHECK_ERROR(hipMemAllocPitch(&ptr, &pitch, maxSizeT, 1, validElementSizeBytes),
                    hipErrorOutOfMemory);
#else
    HIP_CHECK_ERROR(hipMemAllocPitch(&ptr, &pitch, maxSizeT, 1, validElementSizeBytes),
                    hipErrorInvalidValue);
#endif
  }

  SECTION("Max size_t height") {
    HIP_CHECK_ERROR(hipMemAllocPitch(&ptr, &pitch, 1, maxSizeT, validElementSizeBytes),
                    hipErrorOutOfMemory);
  }
}
