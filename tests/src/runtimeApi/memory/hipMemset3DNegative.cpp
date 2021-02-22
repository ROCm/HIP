/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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

 (TestCase 1)::
 1) Test hipMemset3D() with uninitialized devPitchedPtr.
 2) Test hipMemset3DAsync() with uninitialized devPitchedPtr.

 (TestCase 2)::
 3) Reset devPitchedPtr to zero and check return value for hipMemset3D().
 4) Reset devPitchedPtr to zero and check return value for hipMemset3DAsync().

 (TestCase 3)
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

 (TestCase 4)
11) Device Ptr out bound and extent(0) passed for hipMemset3D().
12) Device Ptr out bound and extent(0) passed for hipMemset3DAsync().

 (TestCase 5)
13) Device Ptr out bound and valid extent passed for hipMemset3D().
14) Device Ptr out bound and valid extent passed for hipMemset3DAsync().
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST: %t --tests 1
 * TEST: %t --tests 2
 * TEST: %t --tests 3
 * TEST: %t --tests 4
 * TEST: %t --tests 5
 * HIT_END
 */

#include "test_common.h"

int main(int argc, char *argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  hipStream_t stream;
  hipError_t ret;
  hipPitchedPtr devPitchedPtr;
  bool TestPassed = true;
  int memsetval = 1;
  size_t numH = 256;
  size_t numW = 256;
  size_t depth = 10;
  size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);

  HIPCHECK(hipStreamCreate(&stream));
  HIPCHECK(hipMalloc3D(&devPitchedPtr, extent));

  if (p_tests == 1) {
    // Use uninitialized devpitched ptr
    hipPitchedPtr devPitchedUnPtr;

    if ((ret = hipMemset3D(devPitchedUnPtr, memsetval, extent))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "uninit devpitched ptr. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3DAsync(devPitchedUnPtr, memsetval, extent, stream))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "uninit devpitched ptr(Async). Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }
  } else if (p_tests == 2) {
    // Reset devPitchedPtr to zero
    hipPitchedPtr rdevPitchedPtr = {0};

    if ((ret = hipMemset3D(rdevPitchedPtr, memsetval, extent))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "rdevPitchedPtr(0). Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3DAsync(rdevPitchedPtr, memsetval, extent, stream))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "rdevPitchedPtr(0). Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }
  } else if (p_tests == 3) {
    // Pass extent fields as max size_t
    hipExtent extMW = make_hipExtent(std::numeric_limits<std::size_t>::max(),
                                     numH,
                                     depth);
    hipExtent extMH = make_hipExtent(width,
                                     std::numeric_limits<std::size_t>::max(),
                                     depth);
    hipExtent extMD = make_hipExtent(width,
                                     numH,
                                     std::numeric_limits<std::size_t>::max());

    if ((ret = hipMemset3D(devPitchedPtr, memsetval, extMW))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "extent.width max(size_t). Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3DAsync(devPitchedPtr, memsetval, extMW, stream))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "extent.width max(size_t) Async. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3D(devPitchedPtr, memsetval, extMH))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "extent.height max(size_t). Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3DAsync(devPitchedPtr, memsetval, extMH, stream))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "extent.height max(size_t) Async. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }
#ifdef __HIP_PLATFORM_AMD__
    if ((ret = hipMemset3D(devPitchedPtr, memsetval, extMD))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "extent.depth max(size_t). Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3DAsync(devPitchedPtr, memsetval, extMD, stream))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned for "
             "extent.depth max(size_t) Async. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }
#else
    printf("Cuda doesn't check the maximum depth of extent field\n");
#endif
  } else if (p_tests == 4) {
    // Device Ptr out bound and extent(0) passed for memset

    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * extent.height;

    // Point devptr to end of allocated memory
    char *devPtrMod = (reinterpret_cast<char *>(devPitchedPtr.ptr))
                       + depth * slicePitch;

    // Advance devptr further to go out of boundary
    devPtrMod = devPtrMod + 10;
    hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrMod, pitch,
                                                    numW * sizeof(char), numH);
    hipExtent extent0 = {0};
    if ((ret = hipMemset3D(modDevPitchedPtr, memsetval, extent0))
        != hipSuccess) {
      printf("ArgValidation : Inappropriate error value returned when "
             "deviceptr goes out of boundary. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3DAsync(modDevPitchedPtr, memsetval, extent0, stream))
        != hipSuccess) {
      printf("ArgValidation : Inappropriate error value returned when "
             "deviceptr goes out of boundary Async. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }
  } else if (p_tests == 5) {
    // Device Ptr out bound and valid extent passed for memset

    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * extent.height;

    // Point devptr to end of allocated memory
    char *devPtrMod = (reinterpret_cast<char *>(devPitchedPtr.ptr))
                       + depth * slicePitch;

    // Advance devptr further to go out of boundary
    devPtrMod = devPtrMod + 10;
    hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrMod, pitch,
                                                    numW * sizeof(char), numH);
    if ((ret = hipMemset3D(modDevPitchedPtr, memsetval, extent))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned when "
             "deviceptr goes out of boundary. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }

    if ((ret = hipMemset3DAsync(modDevPitchedPtr, memsetval, extent, stream))
        != hipErrorInvalidValue) {
      printf("ArgValidation : Inappropriate error value returned when "
             "deviceptr goes out of boundary Async. Error: '%s'(%d)\n",
             hipGetErrorString(ret), ret);
      TestPassed &= false;
    }
  } else {
    printf("Didnt receive any valid option. Try options 1 to 5\n");
    TestPassed = false;
  }

  HIPCHECK(hipStreamDestroy(stream));
  HIPCHECK(hipFree(devPitchedPtr.ptr));

  if (TestPassed) {
    passed();
  } else {
    failed("hipMemset3DNegative validation Failed!");
  }
}
