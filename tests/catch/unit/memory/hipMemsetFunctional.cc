/*
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * Testcase Scenarios:
 * For hipMemset, hipMemsetD8, hipMemsetD16, hipMemsetD32, hipMemset2D, hipMemset3D and all async
 * counterparts
 * 1) (ZeroValue)  - Test setting a specified range to zero.
 * 2) (SmallSize)  - Test setting a unique memset value for small sizes.
 * 3) (ZeroSize)   - Test that trying to set memory with a zero dimension does not fail and doesn't
 *                   affect the memory.
 * 4) (PartialSet) - Test setting a partial range of total allocated memory and
 *                   ensure the full range isn't affected.
 */
#include <hip_test_common.hh>

constexpr size_t FULL_DIM = 10;

// Enum used to determine which 1D memset function to use.
enum MemsetType {
  hipMemsetTypeDefault = 1,
  hipMemsetTypeD8 = 2,
  hipMemsetTypeD16 = 3,
  hipMemsetTypeD32 = 4
};

// Macro used to assert all elements in a flat vector range is equal to a specified value
#define HIP_ASSERT_VEC_EQ(ptr, value, N)                                                           \
  for (size_t i = 0; i < N; i++) {                                                                 \
    CAPTURE(N, i, ptr[i], value);                                                                  \
    HIP_ASSERT(ptr[i] == value);                                                                   \
  }

// Copies device data to host and checks that each element is equal to the
// specified value
template <typename T> void check_device_data(T* devPtr, T value, size_t numElems) {
  std::unique_ptr<T[]> hostPtr(new T[numElems]);
  HIP_CHECK(hipMemcpy(hostPtr.get(), devPtr, numElems * sizeof(T), hipMemcpyDeviceToHost));
  HIP_ASSERT_VEC_EQ(hostPtr.get(), value, numElems);
}

// Macro to assist calling and then checking the result of the 1D memset API with the necessary
// manipulation to the arguments.
#define HIP_MEMSET_CHECK(hipMemsetFunc, devPtr, value, count, async)                               \
  using scalar_t = decltype(value);                                                                \
  size_t sizeBytes = count * sizeof(scalar_t);                                                     \
  HIP_CHECK(hipMemsetFunc(devPtr, value, sizeBytes));                                              \
  if (async) {                                                                                     \
    HIP_CHECK(hipStreamSynchronize(stream));                                                       \
  }                                                                                                \
  check_device_data(devPtr, value, count);

#define HIP_MEMSET_CHECK_DTYPE(hipMemsetFunc, devPtr, value, count, async)                         \
  HIP_CHECK(hipMemsetFunc(reinterpret_cast<hipDeviceptr_t>(devPtr), value, count));                \
  if (async) {                                                                                     \
    HIP_CHECK(hipStreamSynchronize(stream));                                                       \
  }                                                                                                \
  check_device_data(devPtr, value, count);

// Enum for specifying wether to allocate the data using hipMalloc, hipHostMalloc or not at all.
enum MemsetMallocType { hipDeviceMalloc_t = 1, hipHostMalloc_t = 2, hipNoMalloc_t };

// Helper function for allocating memory, setting data with the specified 1D memset API and then
// checking result of operation.
template <typename T>
void checkMemset(T value, size_t count, MemsetType memsetType, bool async = false,
                 MemsetMallocType mallocType = hipDeviceMalloc_t, T* devPtr = nullptr) {
  hipStream_t stream{nullptr};
  if (async) {
    hipStreamCreate(&stream);
  }

  // Allocate Memory
  if (mallocType == hipDeviceMalloc_t) {
    HIP_CHECK(hipMalloc(&devPtr, count * sizeof(T)));
  } else if (mallocType == hipHostMalloc_t) {
    HIP_CHECK(hipHostMalloc(&devPtr, count * sizeof(T)));
  }

  // memset API calls
  switch (memsetType) {
    case hipMemsetTypeDefault:
      if (!async) {
        INFO("Testing hipMemset call")
        HIP_MEMSET_CHECK(hipMemset, devPtr, value, count, false);
      } else {
        INFO("Testing hipMemsetAsync call")
        HIP_MEMSET_CHECK(hipMemsetAsync, devPtr, value, count, true);
      }
      break;
    case hipMemsetTypeD8:
      if (!async) {
        INFO("Testing hipMemsetD8 call")
        HIP_MEMSET_CHECK_DTYPE(hipMemsetD8, devPtr, value, count, false);
      } else {
        INFO("Testing hipMemsetD8Async call")
        HIP_MEMSET_CHECK_DTYPE(hipMemsetD8Async, devPtr, value, count, true);
      }
      break;
    case hipMemsetTypeD16:
      if (!async) {
        INFO("Testing hipMemsetD16 call")
        HIP_MEMSET_CHECK_DTYPE(hipMemsetD16, devPtr, value, count, false);
      } else {
        INFO("Testing hipMemsetD16Async call")
        HIP_MEMSET_CHECK_DTYPE(hipMemsetD16Async, devPtr, value, count, true);
      }
      break;
    case hipMemsetTypeD32:
      if (!async) {
        INFO("Testing hipMemsetD32 call")
        HIP_MEMSET_CHECK_DTYPE(hipMemsetD32, devPtr, value, count, false);
      } else {
        INFO("Testing hipMemsetD32Async call")
        HIP_MEMSET_CHECK_DTYPE(hipMemsetD32Async, devPtr, value, count, true);
      }
      break;
  }

  // Cleanup
  if (async) {
    HIP_CHECK(hipStreamDestroy(stream));
  }

  // Free memory
  if (mallocType == hipDeviceMalloc_t) {
    HIP_CHECK(hipFree(devPtr));
  } else if (mallocType == hipHostMalloc_t) {
    HIP_CHECK(hipHostFree(devPtr));
  }
}

// Macro which defines a TEST_CASE which calls and then checks the result of the 1D memset macros
// for all combinations of sync/async and hipMalloc/hipHostMalloc, given the value and memory range.
#define DEFINE_1D_BASIC_TEST_CASE(suffix, memsetType, T, value, count)                             \
  TEST_CASE("Unit_hipMemsetFunctional_" + std::string(suffix)) {                                   \
    const std::string memsetStr = std::string(suffix);                                             \
    SECTION(memsetStr + " - Device Malloc") {                                                      \
      checkMemset(static_cast<T>(value), count, memsetType, false, hipDeviceMalloc_t);             \
    }                                                                                              \
    SECTION(memsetStr + " - Host Malloc") {                                                        \
      checkMemset(static_cast<T>(value), count, memsetType, false, hipHostMalloc_t);               \
    }                                                                                              \
    SECTION(memsetStr + "Async - Device Malloc") {                                                 \
      checkMemset(static_cast<T>(value), count, memsetType, true, hipDeviceMalloc_t);              \
    }                                                                                              \
    SECTION(memsetStr + "Async - Host Malloc") {                                                   \
      checkMemset(static_cast<T>(value), count, memsetType, true, hipHostMalloc_t);                \
    }                                                                                              \
  }

DEFINE_1D_BASIC_TEST_CASE("ZeroValue_hipMemset", hipMemsetTypeDefault, float, 0, 1024)
DEFINE_1D_BASIC_TEST_CASE("ZeroValue_hipMemsetD32", hipMemsetTypeD32, uint32_t, 0, 1024)
DEFINE_1D_BASIC_TEST_CASE("ZeroValue_hipMemsetD16", hipMemsetTypeD16, int16_t, 0, 1024)
DEFINE_1D_BASIC_TEST_CASE("ZeroValue_hipMemsetD8", hipMemsetTypeD8, int8_t, 0, 1024)

DEFINE_1D_BASIC_TEST_CASE("SmallSize_hipMemset", hipMemsetTypeDefault, char, 0x42, 1)
DEFINE_1D_BASIC_TEST_CASE("SmallSize_hipMemsetD32", hipMemsetTypeD32, uint32_t, 0x101, 1)
DEFINE_1D_BASIC_TEST_CASE("SmallSize_hipMemsetD16", hipMemsetTypeD16, int16_t, 0x10, 1)
DEFINE_1D_BASIC_TEST_CASE("SmallSize_hipMemsetD8", hipMemsetTypeD8, int8_t, 0x1, 1)

DEFINE_1D_BASIC_TEST_CASE("ZeroSize_hipMemset", hipMemsetTypeDefault, char, 0x42, 0)
DEFINE_1D_BASIC_TEST_CASE("ZeroSize_hipMemsetD32", hipMemsetTypeD32, uint32_t, 0x101, 0)
DEFINE_1D_BASIC_TEST_CASE("ZeroSize_hipMemsetD16", hipMemsetTypeD16, int16_t, 0x10, 0)
DEFINE_1D_BASIC_TEST_CASE("ZeroSize_hipMemsetD8", hipMemsetTypeD8, int8_t, 0x1, 0)

// Helper function that sets a full region of memory with an initial value, sets a smaller subregion
// with another value and check that the memset API do not write outside of the subregion of data.
template <typename T>
void partialMemsetTest(T valA, T valB, size_t count, size_t offset, MemsetType memsetType,
                       bool async) {
  T* devPtr;
  size_t subSize{count - offset};
  HIP_CHECK(hipMalloc(&devPtr, count * sizeof(T)));

  // Set entire region to be first value.
  INFO("Setting full region");
  checkMemset(valA, count, memsetType, async, hipNoMalloc_t, devPtr);

  // Set partial region to be second value.
  INFO("Setting partial region");
  checkMemset(valB, subSize, memsetType, async, hipNoMalloc_t, devPtr + offset);

  // Ensure the first section remains unchanged
  check_device_data(devPtr, valA, offset);
  HIP_CHECK(hipFree(devPtr));
}

TEST_CASE("Unit_hipMemsetFunctional_PartialSet_1D") {
  for (auto widthOffset = 8; widthOffset <= 8; widthOffset *= 2) {
    SECTION("hipMemset - Partial Set") {
      partialMemsetTest<char>(0x1, 0x42, 1024, widthOffset, hipMemsetTypeDefault, false);
    }
    SECTION("hipMemsetAsync - Partial Set") {
      partialMemsetTest<char>(0x1, 0x42, 1024, widthOffset, hipMemsetTypeDefault, true);
    }
    SECTION("hipMemsetD8 - Partial Set") {
      partialMemsetTest<int8_t>(0x1, 0xDE, 1024, widthOffset, hipMemsetTypeD8, false);
    }
    SECTION("hipMemsetD8Async - Partial Set") {
      partialMemsetTest<int8_t>(0x1, 0xDE, 1024, widthOffset, hipMemsetTypeD8, true);
    }
    SECTION("hipMemsetD16 - Partial Set") {
      partialMemsetTest<int16_t>(0x1, 0xDEAD, 1024, widthOffset, hipMemsetTypeD16, false);
    }
    SECTION("hipMemsetD16Async - Partial Set") {
      partialMemsetTest<int16_t>(0x1, 0xDEAD, 1024, widthOffset, hipMemsetTypeD16, true);
    }
    SECTION("hipMemsetD32 - Partial Set") {
      partialMemsetTest<uint32_t>(0x1, 0xDEADBEEF, 1024, widthOffset, hipMemsetTypeD32, false);
    }
    SECTION("hipMemsetD32Async - Partial Set") {
      partialMemsetTest<uint32_t>(0x1, 0xDEADBEEF, 1024, widthOffset, hipMemsetTypeD32, true);
    }
  }
}

// Helper function that copies the device data to the host and returns a unique_ptr to that data.
template <typename T>
std::unique_ptr<T[]> get_device_data_2D(T* devPtr, size_t pitch, size_t width, size_t height) {
  std::unique_ptr<T[]> hostPtr(new T[width * height]);
  constexpr size_t elementSize = sizeof(T);
  HIP_CHECK(hipMemcpy2D(hostPtr.get(), width * elementSize, devPtr, pitch, width, height,
                        hipMemcpyDeviceToHost));
  return hostPtr;
}

// Copies device data to host and checks that each element is equal to the
// specified value
template <typename T>
void check_device_data_2D(T* devPtr, T value, size_t pitch, size_t width, size_t height) {
  auto hostPtr = get_device_data_2D<T>(devPtr, pitch, width, height);
  HIP_ASSERT_VEC_EQ(hostPtr.get(), value, width * height);
}

// Helper function for allocating memory, setting data with the specified 2D memset API and then
// checking result of operation.
template <typename T>
void checkMemset2D(T value, size_t width, size_t height, bool async = false, size_t pitch = 0,
                   T* devPtr = nullptr) {
  hipStream_t stream{nullptr};
  hipStreamCreate(&stream);
  constexpr size_t elementSize = sizeof(T);
  bool freeDevPtr = false;
  if (devPtr == nullptr) {
    freeDevPtr = true;
    HIP_CHECK(
        hipMallocPitch(reinterpret_cast<void**>(&devPtr), &pitch, width * elementSize, height));
  }

  if (!async) {
    INFO("Testing hipMemset2D call")
    HIP_CHECK(hipMemset2D(devPtr, pitch, value, width * elementSize, height));
  } else {
    INFO("Testing hipMemset2DAsync call")
    HIP_CHECK(hipMemset2DAsync(devPtr, pitch, value, width * elementSize, height, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  if (width * height > 0) {
    check_device_data_2D(devPtr, value, pitch, width, height);
  }
  if (freeDevPtr) {
    HIP_CHECK(hipFree(devPtr));
  }
  hipStreamDestroy(stream);
}

TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_2D") {
  constexpr size_t width{128};
  constexpr size_t height{128};
  constexpr char memsetVal = 0;
  SECTION("hipMemset2D - Zero Value") { checkMemset2D(memsetVal, width, height, false); }
  SECTION("hipMemset2DAsync - Zero Value") { checkMemset2D(memsetVal, width, height, true); }
}

TEST_CASE("Unit_hipMemsetFunctional_SmallSize_2D") {
  constexpr char memsetVal = 0x42;
  SECTION("hipMemset2D - Small Size") { checkMemset2D(memsetVal, 1, 1, false); }
  SECTION("hipMemset2DAsync - Small Size") { checkMemset2D(memsetVal, 1, 1, true); }
}

TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_2D") {
  size_t pitch{0};
  size_t width{10};
  size_t height{10};
  char* devPtr{nullptr};
  HIP_CHECK(
      hipMallocPitch(reinterpret_cast<void**>(&devPtr), &pitch, width * sizeof(char), height));

  const char initValue = 0x1;
  const char testValue = 0x11;
  // Set full region to initial value
  checkMemset2D(initValue, width, height, false, pitch, devPtr);

  SECTION("hipMemset2D - Zero Width") {
    checkMemset2D(testValue, 0, height, false, pitch, devPtr);
    check_device_data_2D(devPtr, initValue, pitch, width, height);
  }
  SECTION("hipMemset2DAsync - Zero Width") {
    checkMemset2D(testValue, 0, height, true, pitch, devPtr);
    check_device_data_2D(devPtr, initValue, pitch, width, height);
  }
  SECTION("hipMemset2D - Zero Height") {
    checkMemset2D(testValue, width, 0, false, pitch, devPtr);
    check_device_data_2D(devPtr, initValue, pitch, width, height);
  }
  SECTION("hipMemset2DAsync - Zero Height") {
    checkMemset2D(testValue, width, 0, true, pitch, devPtr);
    check_device_data_2D(devPtr, initValue, pitch, width, height);
  }
  SECTION("hipMemset2D - Zero Width and Height") {
    checkMemset2D(testValue, 0, 0, false, pitch, devPtr);
    check_device_data_2D(devPtr, initValue, pitch, width, height);
  }
  SECTION("hipMemset2DAsync - Zero Width and Height") {
    checkMemset2D(testValue, 0, 0, true, pitch, devPtr);
    check_device_data_2D(devPtr, initValue, pitch, width, height);
  }
  HIP_CHECK(hipFree(devPtr));
}

// Helper function that sets a full region of memory with an initial value, sets a smaller subregion
// with another value and check that the memset API do not write outside of the subregion of data.
template <typename T>
void partialMemsetTest2D(T valA, T valB, size_t width, size_t height, size_t widthOffset,
                         size_t heightOffset, bool async) {
  T* devPtr{nullptr};
  size_t pitch{0};
  size_t subWidth{width - widthOffset};
  size_t subHeight{height - heightOffset};
  constexpr size_t elementSize = sizeof(T);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devPtr), &pitch, width * elementSize, height));

  // Set entire region to be first value.
  INFO("Setting full square region");
  checkMemset2D(valA, width, height, async, pitch, devPtr);

  // Set partial region to be second value.
  INFO("Setting partial square region")
  checkMemset2D(valB, subWidth, subHeight, async, pitch, devPtr);

  auto hostPtr = get_device_data_2D<T>(devPtr, pitch, width, height);
  T comparVal{0};
  size_t idx{0};
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < height; j++) {
      if (i < subWidth && j < subHeight) {
        // Compare subregion value
        comparVal = valB;
      } else {
        // Compare full region value
        comparVal = valA;
      }
      idx = i * height + j;
      CAPTURE(width, height, subWidth, subHeight, i, j, idx, hostPtr[idx], comparVal);
      HIP_ASSERT(hostPtr[idx] == comparVal);
    }
  }
  HIP_CHECK(hipFree(devPtr));
}

TEST_CASE("Unit_hipMemsetFunctional_PartialSet_2D") {
  for (auto widthOffset = 8; widthOffset <= 128; widthOffset *= 2) {
    for (auto heightOffset = 8; heightOffset <= 128; heightOffset *= 2) {
      SECTION("hipMemset2D - Partial Set") {
        partialMemsetTest2D('a', 'b', 200, 200, widthOffset, heightOffset, false);
      }
      SECTION("hipMemset2DAsync - Partial Set") {
        partialMemsetTest2D('a', 'b', 200, 200, widthOffset, heightOffset, true);
      }
    }
  }
}

// Helper function that copies the device data to the host and returns a unique_ptr to that data.
template <typename T>
std::unique_ptr<T[]> get_device_data_3D(hipPitchedPtr& devPitchedPtr, hipExtent extent) {
  constexpr size_t elementSize = sizeof(T);
  std::unique_ptr<T[]> hostPtr(
      new T[devPitchedPtr.pitch * extent.width * extent.height / elementSize]);
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(hostPtr.get(), devPitchedPtr.pitch,
                                      extent.width / elementSize, extent.height);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&myparms));
  return hostPtr;
}

// Copies device data to host and checks that each element is equal to the
// specified value
template <typename T>
void check_device_data_3D(hipPitchedPtr& devPitchedPtr, T value, hipExtent extent) {
  auto hostPtr = get_device_data_3D<T>(devPitchedPtr, extent);
  size_t width = extent.width / sizeof(T);
  size_t height = extent.height;
  size_t depth = extent.depth;
  size_t idx;
  for (size_t k = 0; k < depth; k++) {
    for (size_t j = 0; j < height; j++) {
      for (size_t i = 0; i < width; i++) {
        idx = devPitchedPtr.pitch * height * k + devPitchedPtr.pitch * j + i;
        INFO("idx=" << idx << " hostPtr[idx]=" << hostPtr[idx] << " value=" << value)
        HIP_ASSERT(hostPtr[idx] == value);
      }
    }
  }
}

// Helper function for allocating memory, setting data with the specified 3D memset API and then
// checking result of operation.
template <typename T>
void checkMemset3D(hipPitchedPtr& devPitchedPtr, T value, hipExtent extent, bool async = false) {
  hipStream_t stream{nullptr};
  hipStreamCreate(&stream);
  if (devPitchedPtr.ptr == nullptr) {
    HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  }
  if (!async) {
    INFO("Testing hipMemset3D call")
    HIP_CHECK(hipMemset3D(devPitchedPtr, value, extent));
  } else {
    INFO("Testing hipMemset3DAsync call")
    HIP_CHECK(hipMemset3DAsync(devPitchedPtr, value, extent, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  if (extent.width * extent.height * extent.depth > 0) {
    check_device_data_3D(devPitchedPtr, value, extent);
  }
  hipStreamDestroy(stream);
}

void check_memset_3D(std::string sectionStr, size_t width, size_t height, size_t depth,
                     char value) {
  hipPitchedPtr devPitchedPtr;
  hipExtent fullExtent;
  constexpr char fullVal = 0x21;
  hipExtent extent = make_hipExtent(width, height, depth);
  // Check if any of the dimensions are zero
  bool anyZero = width * height * depth == 0;
  if (anyZero) {
    // If they are zero then set a full region with memset value to later check if it's changed.
    devPitchedPtr.ptr = nullptr;
    fullExtent = make_hipExtent(FULL_DIM, FULL_DIM, FULL_DIM);
    checkMemset3D(devPitchedPtr, fullVal, fullExtent, false);
  }
  SECTION("hipMemset3D - " + sectionStr) {
    if (!anyZero) {
      devPitchedPtr.ptr = nullptr;
    }
    checkMemset3D(devPitchedPtr, value, extent, false);
    if (anyZero) {
      // Check to make sure memsets with a zero dimension did not affect above set region.
      check_device_data_3D(devPitchedPtr, fullVal, fullExtent);
    }
    HIP_CHECK(hipFree(devPitchedPtr.ptr));
  }
  SECTION("hipMemset3DAsync - " + sectionStr) {
    if (!anyZero) {
      devPitchedPtr.ptr = nullptr;
    }
    checkMemset3D(devPitchedPtr, value, extent, true);
    if (anyZero) {
      // Check to make sure memsets with a zero dimension did not affect above set region.
      check_device_data_3D(devPitchedPtr, fullVal, fullExtent);
    }
    HIP_CHECK(hipFree(devPitchedPtr.ptr));
  }
}

TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_3D") {
  check_memset_3D("Zero Value", 128, 128, 10, 0);
}

TEST_CASE("Unit_hipMemsetFunctional_SmallSize_3D") { check_memset_3D("Small Size", 1, 1, 1, 0x42); }

TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_3D") {
  constexpr size_t elementSize = sizeof(char);
  check_memset_3D("Zero Width", 0, FULL_DIM, FULL_DIM, 0x23);
  check_memset_3D("Zero Height", FULL_DIM * elementSize, 0, FULL_DIM, 0x23);
  check_memset_3D("Zero Depth", FULL_DIM * elementSize, FULL_DIM, 0, 0x23);
  check_memset_3D("Zero Width and Height", 0 * elementSize, 0, FULL_DIM, 0x23);
  check_memset_3D("Zero Width and Depth", 0 * elementSize, FULL_DIM, 0, 0x23);
  check_memset_3D("Zero Height and Depth", FULL_DIM * elementSize, 0, 0, 0x23);
  check_memset_3D("Zero Width, Height and Depth", 0 * elementSize, 0, 0, 0x23);
}

// Helper function that sets a full region of memory with an initial value, sets a smaller subregion
// with another value and check that the memset API do not write outside of the subregion of data.
template <typename T>
void partialMemsetTest3D(T valA, T valB, size_t width, size_t height, size_t depth,
                         size_t widthOffset, size_t heightOffset, size_t depthOffset, bool async) {
  size_t subWidth{width - widthOffset};
  size_t subHeight{height - heightOffset};
  size_t subDepth{depth - depthOffset};
  hipPitchedPtr devPitchedPtr;
  devPitchedPtr.ptr = nullptr;
  hipExtent extent = make_hipExtent(width * sizeof(T), height, depth);
  hipExtent subExtent = make_hipExtent(subWidth * sizeof(T), subHeight, subDepth);

  // Set entire region to be first value.
  INFO("Setting full cuboid region") { checkMemset3D(devPitchedPtr, valA, extent, async); }
  // Set partial region to be second value.
  INFO("Setting partial cuboid region") { checkMemset3D(devPitchedPtr, valB, subExtent, async); }
  auto pitch = devPitchedPtr.pitch;
  auto hostPtr = get_device_data_3D<T>(devPitchedPtr, extent);
  T comparVal{0};
  size_t idx{0};
  for (size_t k = 0; k < depth; k++) {
    for (size_t j = 0; j < height; j++) {
      for (size_t i = 0; i < width; i++) {
        if (i < subWidth && j < subHeight && k < subDepth) {
          comparVal = valB;
        } else {
          comparVal = valA;
        }
        idx = devPitchedPtr.pitch * height * k + devPitchedPtr.pitch * j + i;
        CAPTURE(width, height, depth, pitch, subWidth, subHeight, subDepth, i, j, k, idx,
                hostPtr[idx], comparVal);
        HIP_ASSERT(hostPtr[idx] == comparVal);
      }
    }
  }
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
}

TEST_CASE("Unit_hipMemsetFunctional_PartialSet_3D") {
  for (auto widthOffset = 8; widthOffset <= 128; widthOffset *= 2) {
    for (auto heightOffset = 8; heightOffset <= 128; heightOffset *= 2) {
      for (auto depthOffset = 2; depthOffset <= 5; depthOffset++) {
        SECTION("hipMemset3D - Partial Set") {
          partialMemsetTest3D('a', 'b', 200, 200, 10, widthOffset, heightOffset, depthOffset,
                              false);
        }
        SECTION("hipMemset3DAsync - Partial Set") {
          partialMemsetTest3D('a', 'b', 200, 200, 10, widthOffset, heightOffset, depthOffset, true);
        }
      }
    }
  }
}
