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

/**
 * @brief Test hipMemset Apis using invalid parameters
 *
 */
TEST_CASE("Unit_hipMemset_Negative") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-20");
#endif

  constexpr int Nbytes = validExtent.width * sizeof(char);
  void* dst;

  SECTION("Invalid Dst") {
    SECTION("Uninitialized Dst") {}
    SECTION("Nullptr as dst") { dst = nullptr; }

    std::unique_ptr<char[]> hostPtr;
    SECTION("Host Pointer as Dst") {
      hostPtr.reset(new char[Nbytes]);
      dst = hostPtr.get();
    }

    HIP_CHECK_ERROR(hipMemset(dst, memsetVal, Nbytes), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetAsync(dst, memsetVal, Nbytes, nullptr), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, Nbytes),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, Nbytes, nullptr),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, Nbytes),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, Nbytes, nullptr),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, Nbytes),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, Nbytes, nullptr),
        hipErrorInvalidValue);
  }

#if !HT_AMD /* EXSWCPHIPT-20 */
  SECTION("Out of Bounds Size") {
    constexpr size_t outOfBoundsSize{Nbytes + 1};
    HIP_CHECK(hipMalloc(&dst, Nbytes));

    HIP_CHECK_ERROR(hipMemset(dst, memsetVal, outOfBoundsSize), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetAsync(dst, memsetVal, outOfBoundsSize, nullptr), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, outOfBoundsSize),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal,
                                      outOfBoundsSize, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, outOfBoundsSize),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal,
                                      outOfBoundsSize, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal, outOfBoundsSize),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(dst), memsetVal,
                                     outOfBoundsSize, nullptr),
                    hipErrorInvalidValue);

    HIP_CHECK(hipFree(dst));
  }
#endif

  SECTION("Out of Bounds Ptr") {
    HIP_CHECK(hipMalloc(&dst, Nbytes));
    void* outOfBoundsPtr{reinterpret_cast<char*>(dst) + Nbytes + 1};
    HIP_CHECK_ERROR(hipMemset(outOfBoundsPtr, memsetVal, Nbytes), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetAsync(outOfBoundsPtr, memsetVal, Nbytes), hipErrorInvalidValue);
    HIP_CHECK(hipFree(dst));
  }
}

/**
 * @brief Test hipMemset2D Apis using invalid parameters
 *
 */
TEST_CASE("Unit_hipMemset2D_Negative") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-52");
#endif

  constexpr size_t height = validExtent.height;
  constexpr size_t width = validExtent.width;
  constexpr size_t widthInBytes = width * sizeof(char);

  void* dst;
  SECTION("Invalid Dst") {
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
    HIP_CHECK_ERROR(hipMemset2DAsync(dst, pitch_A, memsetVal, width, height, nullptr),
                    hipErrorInvalidValue);
    hipFree(A_d);
  }

  SECTION("Valid Dst") {
    size_t realPitch;
    HIP_CHECK(hipMallocPitch(&dst, &realPitch, widthInBytes, height));

    SECTION("Invalid Pitch") {
      size_t invalidPitch = 1;

      HIP_CHECK_ERROR(hipMemset2D(dst, invalidPitch, memsetVal, width, height),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemset2DAsync(dst, invalidPitch, memsetVal, width, height, nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid Width") {
      size_t invalidWidth = realPitch + 1;
      HIP_CHECK_ERROR(hipMemset2D(dst, realPitch, memsetVal, invalidWidth, height),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemset2DAsync(dst, realPitch, memsetVal, invalidWidth, height, nullptr),
                      hipErrorInvalidValue);
    }

#if !HT_AMD /* EXSWCPHIPT-52 */
    SECTION("Invalid height") {
      size_t invalidHeight = height + 1;
      HIP_CHECK_ERROR(hipMemset2D(dst, realPitch, memsetVal, width, invalidHeight),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemset2DAsync(dst, realPitch, memsetVal, width, invalidHeight, nullptr),
                      hipErrorInvalidValue);
    }
#endif

    SECTION("Out of Bounds Ptr") {
      HIP_CHECK(hipMallocPitch(&dst, &realPitch, widthInBytes, height));
      void* outOfBoundsPtr{reinterpret_cast<char*>(dst) + realPitch * height + 1};
      HIP_CHECK_ERROR(hipMemset2D(outOfBoundsPtr, realPitch, memsetVal, width, height),
                      hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemset2DAsync(outOfBoundsPtr, realPitch, memsetVal, width, height, nullptr),
          hipErrorInvalidValue);
      HIP_CHECK(hipFree(dst));
    }
  }
}

/**
 * @brief Test hipMemset3D Apis using invalid parameters
 *
 */
TEST_CASE("Unit_hipMemset3D_Negative") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-52");
#endif

  hipPitchedPtr pitchedDevPtr;

  SECTION("Invalid PitchedDevPtr") {
    SECTION("Uninitialized PitchedDevPtr") {}
    SECTION("Zero Initialized PitchedDevPtr") { pitchedDevPtr = {}; }

    HIP_CHECK_ERROR(hipMemset3D(pitchedDevPtr, memsetVal, validExtent), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemset3DAsync(pitchedDevPtr, memsetVal, validExtent, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Modified PitchedDevPtr") {
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
    HIP_CHECK_ERROR(hipMemset3DAsync(pitchedDevPtr, memsetVal, validExtent, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipFree(allocatedMemory));
  }

  SECTION("Valid PitchedDevPtr") {
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
    HIP_CHECK_ERROR(hipMemset3DAsync(pitchedDevPtr, memsetVal, invalidExtent, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipFree(pitchedDevPtr.ptr));
  }

  SECTION("Out of Bounds PitchedDevPtr") {
    HIP_CHECK(hipMalloc3D(&pitchedDevPtr, validExtent));
    hipPitchedPtr outOfBoundsPtr{pitchedDevPtr};
    outOfBoundsPtr.ptr = reinterpret_cast<char*>(pitchedDevPtr.ptr) +
        pitchedDevPtr.pitch * validExtent.height * validExtent.depth + 1;

    SECTION("Extent Equal to 0") {
      hipExtent zeroExtent{0, 0, 0};
      HIP_CHECK(hipMemset3D(outOfBoundsPtr, memsetVal, zeroExtent));
      HIP_CHECK(hipMemset3DAsync(outOfBoundsPtr, memsetVal, zeroExtent, nullptr));
    }

    SECTION("Valid Extent") {
      HIP_CHECK_ERROR(hipMemset3D(outOfBoundsPtr, memsetVal, validExtent), hipErrorInvalidValue);
      HIP_CHECK_ERROR(hipMemset3DAsync(outOfBoundsPtr, memsetVal, validExtent, nullptr),
                      hipErrorInvalidValue);
    }
  }
}
