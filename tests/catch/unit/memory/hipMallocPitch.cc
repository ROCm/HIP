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
  MemoryInfo memAfterAllocation{createMemoryInfo()};
  const size_t theoreticalAllocatedMemory{pitch * extent.height * extent.depth};
  const size_t allocatedMemory = memBeforeAllocation.freeMem - memAfterAllocation.freeMem;

  if (theoreticalAllocatedMemory == 0) {
    REQUIRE(theoreticalAllocatedMemory == allocatedMemory);
  } else {
    REQUIRE(theoreticalAllocatedMemory <= allocatedMemory);
  }

  void* hostPtr = std::malloc(theoreticalAllocatedMemory);
  std::memset(hostPtr, 2, theoreticalAllocatedMemory);

  hipPitchedPtr devPitchedPtr{devPtr, pitch, extent.width, extent.height};
  hipPitchedPtr hostPitchedPtr{hostPtr, pitch, extent.width, extent.height};

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
        char* reinterpretedPtr = reinterpret_cast<char*>(hostPtr);
        size_t index = (pitch * extent.height * depth) + (pitch * height) + width;
        if (*(reinterpretedPtr + index) != 1) {
          mismatch = true;
        }
      }
    }
  }
  INFO("Width: " << extent.width << " Height: " << extent.height << " Depth: " << extent.depth);
  REQUIRE(!mismatch);

  free(hostPtr);
}

class ExtentGenerator {
 public:
  static constexpr size_t minWidth{1};
  static constexpr size_t maxWidth{1024};
  static constexpr size_t minHeight{1};
  static constexpr size_t maxHeight{100};
  static constexpr size_t minDepth{1};
  static constexpr size_t maxDepth{100};
  static constexpr size_t totalRandomValues{20};
  static constexpr size_t seed{1337};

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
      extents3D.push_back(hipExtent{minWidth + randomGenerator() % maxWidth,
                                    minHeight + randomGenerator() % maxHeight,
                                    minDepth + randomGenerator() % maxDepth});
    }

    extents2D = std::vector<hipExtent>{hipExtent{0, 0, 1}, hipExtent{1, 0, 1}, hipExtent{0, 1, 1}};

    for (size_t i = 0; i < totalRandomValues; ++i) {
      extents2D.push_back(hipExtent{minWidth + randomGenerator() % maxWidth,
                                    minHeight + randomGenerator() % maxHeight, 1});
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
  HIP_CHECK(
      hipMemAllocPitch(&ptr, &pitch, validExtent.width, validExtent.height, elementSizeBytes));
  validateMemory(ptr, validExtent, pitch, memBeforeAllocation);
  HIP_CHECK(hipFree(ptr));
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
  hipExtent validExtent{1, 1, 1};
  HIP_CHECK_ERROR(hipMalloc3D(nullptr, validExtent), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipMallocPitch_Negative") {
  size_t pitch;
  HIP_CHECK_ERROR(hipMallocPitch(nullptr, &pitch, 1, 1), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipMemAllocPitch_Negative") {
  size_t pitch;
  unsigned int validElementSizeBytes{4};
  HIP_CHECK_ERROR(hipMemAllocPitch(nullptr, &pitch, 1, 1, validElementSizeBytes),
                  hipErrorInvalidValue);
}
