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

#include <hip_test_common.hh>
#include <limits>
#include "hip/driver_types.h"

static constexpr size_t memsetVal{0x42};
static constexpr hipExtent validExtent{184, 57, 16};
static constexpr size_t height{validExtent.height};
static constexpr size_t width{validExtent.width};
static constexpr int widthInBytes = validExtent.width * sizeof(char);
static constexpr hipStream_t nullStream{nullptr};


TEST_CASE("Unit_hipMemset_Negative_InvalidPtr") {
  void* dst;

  SECTION("Uninitialized Dst") {}
  SECTION("Nullptr as dst") { dst = nullptr; }

  std::unique_ptr<char[]> hostPtr;
  SECTION("Host Pointer as Dst") {
    hostPtr.reset(new char[widthInBytes]);
    dst = hostPtr.get();
  }

  HIP_CHECK_ERROR(hipMemset(dst, memsetVal, widthInBytes), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetAsync(dst, memsetVal, widthInBytes, nullStream), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, widthInBytes),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(
      hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, widthInBytes, nullStream),
      hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, widthInBytes),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(
      hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, widthInBytes, nullStream),
      hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, widthInBytes),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(
      hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, widthInBytes, nullStream),
      hipErrorInvalidValue);
}


TEST_CASE("Unit_hipMemset_Negative_OutOfBoundsSize") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-20");
#endif

#if !HT_AMD
  void* dst;
  constexpr size_t outOfBoundsSize{widthInBytes + 1};
  HIP_CHECK(hipMalloc(&dst, widthInBytes));

  HIP_CHECK_ERROR(hipMemset(dst, memsetVal, outOfBoundsSize), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetAsync(dst, memsetVal, outOfBoundsSize, nullStream),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, outOfBoundsSize),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal,
                                    outOfBoundsSize, nullStream),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, outOfBoundsSize),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal,
                                    outOfBoundsSize, nullStream),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, outOfBoundsSize),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal,
                                   outOfBoundsSize, nullStream),
                  hipErrorInvalidValue);

  HIP_CHECK(hipFree(dst));
#endif
}

TEST_CASE("Unit_hipMemset_Negative_OutOfBoundsPtr") {
  void* dst;
  HIP_CHECK(hipMalloc(&dst, widthInBytes));
  void* outOfBoundsPtr{reinterpret_cast<char*>(dst) + widthInBytes + 1};
  HIP_CHECK_ERROR(hipMemset(outOfBoundsPtr, memsetVal, widthInBytes), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemsetAsync(outOfBoundsPtr, memsetVal, widthInBytes, nullStream),
                  hipErrorInvalidValue);
  HIP_CHECK(hipFree(dst));
}

TEST_CASE("Unit_hipMemset2D_Negative_InvalidPtr") {
  void* dst;
  SECTION("Uninitialized Dst") {}
  SECTION("Nullptr as Dst") { dst = nullptr; }

  std::unique_ptr<char[]> hostPtr;
  SECTION("Host Pointer as Dst") {
    hostPtr.reset(new char[height * width]);
    dst = hostPtr.get();
  }

  void* A_d;
  size_t pitch_A;
  HIP_CHECK(hipMallocPitch(&A_d, &pitch_A, widthInBytes, height));
  HIP_CHECK_ERROR(hipMemset2D(dst, pitch_A, memsetVal, width, height), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemset2DAsync(dst, pitch_A, memsetVal, width, height, nullStream),
                  hipErrorInvalidValue);
  hipFree(A_d);
}

TEST_CASE("Unit_hipMemset2D_Negative_InvalidSizes") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-52");
#endif

  void* dst;
  size_t realPitch;
  HIP_CHECK(hipMallocPitch(&dst, &realPitch, widthInBytes, height));

  SECTION("Invalid Pitch") {
    size_t invalidPitch = 1;

    HIP_CHECK_ERROR(hipMemset2D(dst, invalidPitch, memsetVal, width, height), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemset2DAsync(dst, invalidPitch, memsetVal, width, height, nullStream),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Width") {
    size_t invalidWidth = realPitch + 1;
    HIP_CHECK_ERROR(hipMemset2D(dst, realPitch, memsetVal, invalidWidth, height),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemset2DAsync(dst, realPitch, memsetVal, invalidWidth, height, nullStream),
                    hipErrorInvalidValue);
  }

#if !HT_AMD /* EXSWCPHIPT-52 */
  SECTION("Invalid height") {
    size_t invalidHeight = height + 1;
    HIP_CHECK_ERROR(hipMemset2D(dst, realPitch, memsetVal, width, invalidHeight),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemset2DAsync(dst, realPitch, memsetVal, width, invalidHeight, nullStream),
                    hipErrorInvalidValue);
  }
#endif
  HIP_CHECK(hipFree(dst));
}

TEST_CASE("Unit_hipMemset2D_Negative_OutOfBoundsPtr") {
  void* dst;
  size_t realPitch;

  HIP_CHECK(hipMallocPitch(&dst, &realPitch, widthInBytes, height));
  void* outOfBoundsPtr{reinterpret_cast<char*>(dst) + realPitch * height + 1};
  HIP_CHECK_ERROR(hipMemset2D(outOfBoundsPtr, realPitch, memsetVal, width, height),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemset2DAsync(outOfBoundsPtr, realPitch, memsetVal, width, height, nullStream),
                  hipErrorInvalidValue);

  HIP_CHECK(hipFree(dst));
}


TEST_CASE("Unit_hipMemset3D_Negative_InvalidPtr") {
  hipPitchedPtr pitchedDevPtr;

  SECTION("Uninitialized PitchedDevPtr") {}
  SECTION("Zero Initialized PitchedDevPtr") { pitchedDevPtr = {}; }

  HIP_CHECK_ERROR(hipMemset3D(pitchedDevPtr, memsetVal, validExtent), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemset3DAsync(pitchedDevPtr, memsetVal, validExtent, nullStream),
                  hipErrorInvalidValue);
}

TEST_CASE("Unit_hipMemset3D_Negative_ModifiedPtr") {
  hipPitchedPtr pitchedDevPtr;

  HIP_CHECK(hipMalloc3D(&pitchedDevPtr, validExtent));
  void* allocatedMemory{pitchedDevPtr.ptr};

  SECTION("Nullptr Dst") { pitchedDevPtr.ptr = nullptr; }

  std::unique_ptr<char[]> hostPtr;
  SECTION("Host Pointer as Dst") {
    hostPtr.reset(new char[validExtent.width * validExtent.height * validExtent.depth]);
    pitchedDevPtr.ptr = hostPtr.get();
  }

  SECTION("Invalid Pitch") { pitchedDevPtr.pitch = 1; }

  CAPTURE(pitchedDevPtr.ptr, pitchedDevPtr.pitch, pitchedDevPtr.xsize, pitchedDevPtr.ysize);
  HIP_CHECK_ERROR(hipMemset3D(pitchedDevPtr, memsetVal, validExtent), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemset3DAsync(pitchedDevPtr, memsetVal, validExtent, nullStream),
                  hipErrorInvalidValue);
  HIP_CHECK(hipFree(allocatedMemory));
}

TEST_CASE("Unit_hipMemset3D_Negative_InvalidSizes") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-52");
#endif

  hipPitchedPtr pitchedDevPtr;
  HIP_CHECK(hipMalloc3D(&pitchedDevPtr, validExtent));
  hipExtent invalidExtent{validExtent};

  SECTION("Max Width") { invalidExtent.width = std::numeric_limits<std::size_t>::max(); }

  SECTION("Max Height") { invalidExtent.height = std::numeric_limits<std::size_t>::max(); }

#if !HT_NVIDIA /* This case hangs on Nvidia */
  SECTION("Max Depth") { invalidExtent.depth = std::numeric_limits<std::size_t>::max(); }
#endif

  SECTION("Invalid Width") { invalidExtent.width = pitchedDevPtr.pitch + 1; }

#if !HT_AMD /* EXSWCPHIPT-52 */
  SECTION("Invalid height") { invalidExtent.height += 1; }

  SECTION("Invalid depth") { invalidExtent.depth += 1; }
#endif

  CAPTURE(invalidExtent.width, invalidExtent.height, invalidExtent.depth);
  HIP_CHECK_ERROR(hipMemset3D(pitchedDevPtr, memsetVal, invalidExtent), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemset3DAsync(pitchedDevPtr, memsetVal, invalidExtent, nullStream),
                  hipErrorInvalidValue);
  HIP_CHECK(hipFree(pitchedDevPtr.ptr));
}

TEST_CASE("Unit_hipMemset3D_Negative_OutOfBounds") {
  hipPitchedPtr pitchedDevPtr;

  HIP_CHECK(hipMalloc3D(&pitchedDevPtr, validExtent));
  hipPitchedPtr outOfBoundsPtr{pitchedDevPtr};
  outOfBoundsPtr.ptr = reinterpret_cast<char*>(pitchedDevPtr.ptr) +
      pitchedDevPtr.pitch * validExtent.height * validExtent.depth + 1;

  SECTION("Extent Equal to 0") {
    hipExtent zeroExtent{0, 0, 0};
    HIP_CHECK(hipMemset3D(outOfBoundsPtr, memsetVal, zeroExtent));
    HIP_CHECK(hipMemset3DAsync(outOfBoundsPtr, memsetVal, zeroExtent, nullStream));
  }

  SECTION("Valid Extent") {
    HIP_CHECK_ERROR(hipMemset3D(outOfBoundsPtr, memsetVal, validExtent), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemset3DAsync(outOfBoundsPtr, memsetVal, validExtent, nullStream),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(pitchedDevPtr.ptr));
}
