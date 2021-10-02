/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
 1) Test hipMemset3D() with uninitialized devPitchedPtr.
 2) Test hipMemset3DAsync() with uninitialized devPitchedPtr.

 3) Reset devPitchedPtr to zero and check return value for hipMemset3D().
 4) Reset devPitchedPtr to zero and check return value for hipMemset3DAsync().

 5) Test hipMemset3D() with extent.width as max size_t and keeping height,
 depth as valid values.
 6) Test hipMemset3DAsync() with extent.width as max size_t and keeping height,
 depth as valid values.
 7) Test hipMemset3D() with extent.height as max size_t and keeping width,
 depth as valid values.
 8) Test hipMemset3DAsync() with extent.height as max size_t and keeping width,
 depth as valid values.
 9) Test hipMemset3D() with extent.depth as max size_t and keeping height,
 width as valid values.
10) Test hipMemset3DAsync() with extent.depth as max size_t and keeping height,
 width as valid values.

11) Device Ptr out bound and extent(0) passed for hipMemset3D().
12) Device Ptr out bound and extent(0) passed for hipMemset3DAsync().

13) Device Ptr out bound and valid extent passed for hipMemset3D().
14) Device Ptr out bound and valid extent passed for hipMemset3DAsync().
*/

#include <hip_test_common.hh>

TEST_CASE("Unit_hipMemset3D_Negative") {
  hipError_t ret;
  hipPitchedPtr devPitchedPtr;
  constexpr int memsetval = 1;
  constexpr size_t numH = 256, numW = 256;
  constexpr size_t depth = 10;
  constexpr size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));

  SECTION("Using uninitialized devpitched ptr") {
    hipPitchedPtr devPitchedUnPtr;

    ret = hipMemset3D(devPitchedUnPtr, memsetval, extent);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Reset devPitchedPtr to zero") {
    hipPitchedPtr rdevPitchedPtr{};

    ret = hipMemset3D(rdevPitchedPtr, memsetval, extent);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Pass extent fields as max size_t") {
    hipExtent extMW = make_hipExtent(std::numeric_limits<std::size_t>::max(),
                                     numH,
                                     depth);
    hipExtent extMH = make_hipExtent(width,
                                     std::numeric_limits<std::size_t>::max(),
                                     depth);
    hipExtent extMD = make_hipExtent(width,
                                     numH,
                                     std::numeric_limits<std::size_t>::max());

    ret = hipMemset3D(devPitchedPtr, memsetval, extMW);
    REQUIRE(ret != hipSuccess);

    ret = hipMemset3D(devPitchedPtr, memsetval, extMH);
    REQUIRE(ret != hipSuccess);

    if ((TestContext::get()).isAmd()) {
      ret = hipMemset3D(devPitchedPtr, memsetval, extMD);
      REQUIRE(ret != hipSuccess);
    } else {
      WARN("Test is skipped for max depth."
           << "Cuda doesn't check the maximum depth of extent field");
    }
  }

  SECTION("Device Ptr out bound and extent(0) passed for memset") {
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * extent.height;
    constexpr auto advanceOffset = 10;

    // Point devptr to end of allocated memory
    char *devPtrMod = (reinterpret_cast<char *>(devPitchedPtr.ptr))
                       + depth * slicePitch;

    // Advance devptr further to go out of boundary
    devPtrMod = devPtrMod + advanceOffset;
    hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrMod, pitch,
                                                    numW * sizeof(char), numH);
    hipExtent extent0{};
    ret = hipMemset3D(modDevPitchedPtr, memsetval, extent0);

    // api expected to check extent0 and return success before going for ptr.
    REQUIRE(ret == hipSuccess);
  }

  SECTION("Device Ptr out bound and valid extent passed for memset") {
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * extent.height;
    constexpr auto advanceOffset = 10;

    // Point devptr to end of allocated memory
    char *devPtrMod = (reinterpret_cast<char *>(devPitchedPtr.ptr))
                       + depth * slicePitch;

    // Advance devptr further to go out of boundary
    devPtrMod = devPtrMod + advanceOffset;
    hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrMod, pitch,
                                                    numW * sizeof(char), numH);
    ret = hipMemset3D(modDevPitchedPtr, memsetval, extent);

    REQUIRE(ret != hipSuccess);
  }

  HIP_CHECK(hipFree(devPitchedPtr.ptr));
}

TEST_CASE("Unit_hipMemset3DAsync_Negative") {
  hipError_t ret;
  hipPitchedPtr devPitchedPtr;
  hipStream_t stream;
  constexpr int memsetval = 1;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  constexpr size_t depth = 10;
  constexpr size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));

  SECTION("Using uninitialized devpitched ptr") {
    hipPitchedPtr devPitchedUnPtr;

    ret = hipMemset3DAsync(devPitchedUnPtr, memsetval, extent, stream);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Reset devPitchedPtr to zero") {
    hipPitchedPtr rdevPitchedPtr{};

    ret = hipMemset3DAsync(rdevPitchedPtr, memsetval, extent, stream);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Pass extent fields as max size_t") {
    hipExtent extMW = make_hipExtent(std::numeric_limits<std::size_t>::max(),
                                     numH,
                                     depth);
    hipExtent extMH = make_hipExtent(width,
                                     std::numeric_limits<std::size_t>::max(),
                                     depth);
    hipExtent extMD = make_hipExtent(width,
                                     numH,
                                     std::numeric_limits<std::size_t>::max());

    ret = hipMemset3DAsync(devPitchedPtr, memsetval, extMW, stream);
    REQUIRE(ret != hipSuccess);

    ret = hipMemset3DAsync(devPitchedPtr, memsetval, extMH, stream);
    REQUIRE(ret != hipSuccess);

    if ((TestContext::get()).isAmd()) {
      ret = hipMemset3DAsync(devPitchedPtr, memsetval, extMD, stream);
      REQUIRE(ret != hipSuccess);
    } else {
      WARN("Test is skipped for max depth."
           << "Cuda doesn't check the maximum depth of extent field");
    }
  }

  SECTION("Device Ptr out bound and extent(0) passed for memset") {
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * extent.height;
    constexpr auto advanceOffset = 10;

    // Point devptr to end of allocated memory
    char *devPtrMod = (reinterpret_cast<char *>(devPitchedPtr.ptr))
                       + depth * slicePitch;

    // Advance devptr further to go out of boundary
    devPtrMod = devPtrMod + advanceOffset;
    hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrMod, pitch,
                                                    numW * sizeof(char), numH);
    hipExtent extent0{};
    ret = hipMemset3DAsync(modDevPitchedPtr, memsetval, extent0, stream);

    // api expected to check extent0 and return success before going for ptr.
    REQUIRE(ret == hipSuccess);
  }

  SECTION("Device Ptr out bound and valid extent passed for memset") {
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * extent.height;
    constexpr auto advanceOffset = 10;

    // Point devptr to end of allocated memory
    char *devPtrMod = (reinterpret_cast<char *>(devPitchedPtr.ptr))
                       + depth * slicePitch;

    // Advance devptr further to go out of boundary
    devPtrMod = devPtrMod + advanceOffset;
    hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrMod, pitch,
                                                    numW * sizeof(char), numH);
    ret = hipMemset3DAsync(modDevPitchedPtr, memsetval, extent, stream);

    REQUIRE(ret != hipSuccess);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
}
