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
 1) Passing width as 0 in extent, verify hipMemset3D api returns success and
 doesn't modify the buffer passed.
 2) Passing width as 0 in extent, verify hipMemset3DAsync api returns success
 and doesn't modify the buffer passed.

 3) Passing height as 0 in extent, verify hipMemset3D api returns success and
 doesn't modify the buffer passed.
 4) Passing height as 0 in extent, verify hipMemset3DAsync api returns success
 and doesn't modify the buffer passed.

 5) Passing depth as 0 in extent, verify hipMemset3D api returns success and
 doesn't modify the buffer passed.
 6) Passing depth as 0 in extent, verify hipMemset3DAsync api returns success
 and doesn't modify the buffer passed.

 7) When extent passed with width, height and depth all as zeroes, verify
 hipMemset3D api returns success and doesn't modify the buffer passed.
 8) When extent passed with width, height and depth all as zeroes, verify
 hipMemset3DAsync api returns success and doesn't modify the buffer passed.

 (TestCase 2)::
 9) Validate data after performing memory set operation with max memset value
 for hipMemset3D api.
 10) Validate data after performing memory set operation with max memset value
 for hipMemset3DAsync api.

 (TestCase 3)::
 11) Select random slice of 3d array and Memset complete slice with
 hipMemset3D api.
 12) Select random slice of 3d array and Memset complete slice with
 hipMemset3DAsync api.

 (TestCase 4)::
 13) Seek device pitched ptr to desired portion of 3d array and memset the
 portion with hipMemset3D api.
 14) Seek device pitched ptr to desired portion of 3d array and memset the
 portion with hipMemset3DAsync api.

*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST: %t --tests 1
 * TEST: %t --tests 2
 * TEST: %t --tests 3
 * TEST: %t --tests 4
 * HIT_END
 */

#include "test_common.h"

/*
 * Defines
 */
#define MEMSETVAL 1
#define TESTVAL 2
#define NUMH_EXT 256
#define NUMW_EXT 100
#define DEPTH_EXT 10
#define NUMH_MAX 256
#define NUMW_MAX 256
#define DEPTH_MAX 10
#define ZSIZE_S 32
#define YSIZE_S 32
#define XSIZE_S 32
#define ZSIZE_P 30
#define YSIZE_P 30
#define XSIZE_P 30
#define ZPOS_START 10
#define ZSET_LEN 10
#define ZPOS_END 19
#define YPOS_START 10
#define YSET_LEN 10
#define YPOS_END 19
#define XPOS_START 10
#define XSET_LEN 10
#define XPOS_END 19


/**
 * Memset with extent passed and verify data to be intact
 */
bool testMemsetWithExtent(bool bAsync, hipExtent tstExtent) {
  hipPitchedPtr devPitchedPtr;
  bool testPassed = true;
  hipError_t ret;
  char *A_h;
  size_t numH = NUMH_EXT, numW = NUMW_EXT, depth = DEPTH_EXT;
  size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);

  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;

  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);
  memset(A_h, 0, sizeElements);

  HIPCHECK(hipMalloc3D(&devPitchedPtr, extent));
  if (bAsync) {
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    if ((ret = hipMemset3DAsync(devPitchedPtr, MEMSETVAL, extent, stream))
        != hipSuccess) {
      printf("testMemsetWithExtent(%zu,%zu,%zu) Async: Expected to return"
             " success but returned Error: '%s'(%d)\n", extent.width,
             extent.height, extent.depth, hipGetErrorString(ret), ret);
      testPassed &= false;
    }

    if ((ret = hipMemset3DAsync(devPitchedPtr, TESTVAL, tstExtent, stream))
        != hipSuccess) {
      printf("testMemsetWithExtent(%zu,%zu,%zu) Async: Expected to return"
             " success but returned Error: '%s'(%d)\n", tstExtent.width,
             tstExtent.height, tstExtent.depth, hipGetErrorString(ret), ret);
      testPassed &= false;
    }

    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipStreamDestroy(stream));
  } else {
    if ((ret = hipMemset3D(devPitchedPtr, MEMSETVAL, extent))
        != hipSuccess) {
      printf("testMemsetWithExtent(%zu,%zu,%zu) : Expected to return"
             " success but returned Error: '%s'(%d)\n", extent.width,
             extent.height, extent.depth, hipGetErrorString(ret), ret);
      testPassed &= false;
    }

    if ((ret = hipMemset3D(devPitchedPtr, TESTVAL, tstExtent))
        != hipSuccess) {
      printf("testMemsetWithExtent(%zu,%zu,%zu) : Expected to return"
             " success but returned Error: '%s'(%d)\n", tstExtent.width,
             tstExtent.height, tstExtent.depth, hipGetErrorString(ret), ret);
      testPassed &= false;
    }
  }

  if (testPassed) {
    hipMemcpy3DParms myparms = {0};
    myparms.srcPos = make_hipPos(0, 0, 0);
    myparms.dstPos = make_hipPos(0, 0, 0);
    myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
    myparms.srcPtr = devPitchedPtr;
    myparms.extent = extent;
  #ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
  #else
    myparms.kind = hipMemcpyDeviceToHost;
  #endif

    HIPCHECK(hipMemcpy3D(&myparms));

    for (int i = 0; i < elements; i++) {
      if (A_h[i] != MEMSETVAL) {
        testPassed = false;
        printf("testMemsetWithExtent: mismatch at index:%d computed:%02x, "
               "memsetval:%02x\n", i, static_cast<int>(A_h[i]),
               static_cast<int>(MEMSETVAL));
        break;
      }
    }
  }

  HIPCHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
  return testPassed;
}

/**
 * Validates data after performing memory set operation with max memset value
 */
bool testMemsetMaxValue(bool bAsync) {
  hipPitchedPtr devPitchedPtr;
  bool testPassed = true;
  unsigned char *A_h;
  int memsetval = std::numeric_limits<unsigned char>::max();
  size_t numH = NUMH_MAX, numW = NUMW_MAX, depth = DEPTH_MAX;
  size_t width = numW * sizeof(unsigned char);
  hipExtent extent = make_hipExtent(width, numH, depth);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;

  A_h = reinterpret_cast<unsigned char *> (malloc(sizeElements));
  HIPASSERT(A_h != NULL);
  memset(A_h, 0, sizeElements);

  HIPCHECK(hipMalloc3D(&devPitchedPtr, extent));
  if (bAsync) {
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipStreamDestroy(stream));
  } else {
    HIPCHECK(hipMemset3D(devPitchedPtr, memsetval, extent));
  }

  hipMemcpy3DParms myparms = {0};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIPCHECK(hipMemcpy3D(&myparms));

  for (int i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      testPassed = false;
      printf("testMemsetMaxValue: mismatch at index:%d computed:%02x, "
             "memsetval:%02x\n", i, static_cast<int>(A_h[i]),
             static_cast<int>(memsetval));
      break;
    }
  }
  HIPCHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
  return testPassed;
}

/**
 * Function seeks device ptr to random slice and performs Memset operation
 * on the slice selected.
 */
bool seekAndSet3DArraySlice(bool bAsync) {
  char array3D[ZSIZE_S][YSIZE_S][XSIZE_S] = {0};
  bool testPassed = true;
  dim3 arr_dimensions = dim3(ZSIZE_S, YSIZE_S, XSIZE_S);
  hipExtent extent = make_hipExtent(sizeof(char) * arr_dimensions.x,
                                    arr_dimensions.y, arr_dimensions.z);
  hipPitchedPtr devicePitchedPointer;
  int memsetval = MEMSETVAL, memsetval4seeked = TESTVAL;

  HIPCHECK(hipMalloc3D(&devicePitchedPointer, extent));
  HIPCHECK(hipMemset3D(devicePitchedPointer, memsetval, extent));

  // select random slice for memset
  unsigned int seed = time(NULL);
  int slice_index = rand_r(&seed) % ZSIZE_S;

  printf("memset3d for sliceindex %d\n", slice_index);

  // Get attributes from device pitched pointer
  size_t pitch = devicePitchedPointer.pitch;
  size_t slicePitch = pitch * extent.height;

  // Point devptr to selected slice
  char *devPtrSlice = (reinterpret_cast<char *>(devicePitchedPointer.ptr))
                       + slice_index * slicePitch;
  hipExtent extentSlice = make_hipExtent(sizeof(char) * arr_dimensions.x,
                                         arr_dimensions.y, 1);
  hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrSlice, pitch,
                                         arr_dimensions.x, arr_dimensions.y);

  if (bAsync) {
    // Memset selected slice (Async)
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipMemset3DAsync(modDevPitchedPtr, memsetval4seeked,
                              extentSlice, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipStreamDestroy(stream));
  } else {
    // Memset selected slice
    HIPCHECK(hipMemset3D(modDevPitchedPtr, memsetval4seeked, extentSlice));
  }

  // Copy result back to host buffer
  hipMemcpy3DParms myparms = {0};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(array3D, sizeof(char) * arr_dimensions.x,
                                      arr_dimensions.x, arr_dimensions.y);
  myparms.srcPtr = devicePitchedPointer;
  myparms.extent = extent;
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIPCHECK(hipMemcpy3D(&myparms));

  for (int z = 0; z < ZSIZE_S; z++) {
    for (int y = 0; y < YSIZE_S; y++) {
      for (int x = 0; x < XSIZE_S; x++) {
        if (z == slice_index) {
          if (array3D[z][y][x] != memsetval4seeked) {
            testPassed = false;
            printf("seekAndSet3DArray Slice: mismatch at index: Arr(%d,%d,%d)"
                   " computed:%02x, memsetval:%02x\n", z, y, x,
                   array3D[z][y][x], memsetval4seeked);
            break;
          }
        } else {
          if (array3D[z][y][x] != memsetval) {
            testPassed = false;
            printf("seekAndSet3DArray Slice: mismatch at index: Arr(%d,%d,%d)"
                   " computed:%02x, memsetval:%02x\n", z, y, x,
                   array3D[z][y][x], memsetval);
            break;
          }
        }
      }
    }
  }

  HIPCHECK(hipFree(devicePitchedPointer.ptr));
  return testPassed;
}

/**
 * Function seeks device ptr to selected portion of 3d array
 * and performs Memset operation on the portion.
 */
bool seekAndSet3DArrayPortion(bool bAsync) {
  char array3D[ZSIZE_P][YSIZE_P][XSIZE_P] = {0};
  bool testPassed = true;
  dim3 arr_dimensions = dim3(ZSIZE_P, YSIZE_P, XSIZE_P);
  hipExtent extent = make_hipExtent(sizeof(char) * arr_dimensions.x,
                                    arr_dimensions.y, arr_dimensions.z);
  hipPitchedPtr devicePitchedPointer;
  int memsetval = MEMSETVAL, memsetval4seeked = TESTVAL;

  HIPCHECK(hipMalloc3D(&devicePitchedPointer, extent));
  HIPCHECK(hipMemset3D(devicePitchedPointer, memsetval, extent));

  // For memsetting extent/size(10,10,10) in the mid portion of cube(30,30,30),
  // seek device ptr to (10,10,10) and then memset 10 bytes across x,y,z axis.
  size_t pitch = devicePitchedPointer.pitch;
  size_t slicePitch = pitch * extent.height;
  int slice_index = ZPOS_START, y = YPOS_START, x = XPOS_START;

  // Select 10th slice
  char *devPtrSlice = (reinterpret_cast<char *>(devicePitchedPointer.ptr))
                       + slice_index * slicePitch;

  // Now select row at height as 10
  char *current_row = reinterpret_cast<char *>(devPtrSlice + y * pitch);

  // Now select index of selected row as 10
  char *devPtrIndexed = &current_row[x];

  // Make dev Pitchedptr, extent
  hipPitchedPtr modDevPitchedPtr = make_hipPitchedPtr(devPtrIndexed, pitch,
                                         arr_dimensions.x, arr_dimensions.y);
  hipExtent setExtent = make_hipExtent(sizeof(char) * XSET_LEN, YSET_LEN,
                                       ZSET_LEN);

  if (bAsync) {
    // Memset selected portion (Async)
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipMemset3DAsync(modDevPitchedPtr, memsetval4seeked,
                              setExtent, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipStreamDestroy(stream));
  } else {
    // Memset selected portion
    HIPCHECK(hipMemset3D(modDevPitchedPtr, memsetval4seeked, setExtent));
  }

  // Copy result back to host buffer
  hipMemcpy3DParms myparms = {0};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(array3D, sizeof(char) * arr_dimensions.x,
                                      arr_dimensions.x, arr_dimensions.y);
  myparms.srcPtr = devicePitchedPointer;
  myparms.extent = extent;
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIPCHECK(hipMemcpy3D(&myparms));

  for (int z = 0; z < ZSIZE_P; z++) {
    for (int y = 0; y < YSIZE_P; y++) {
      for (int x = 0; x < XSIZE_P; x++) {
        if ((z >= ZPOS_START && z <= ZPOS_END) &&
            (y >= YPOS_START && y <= YPOS_END) &&
            (x >= XPOS_START && x <= XPOS_END)) {
          if (array3D[z][y][x] != memsetval4seeked) {
            testPassed = false;
            printf("seekAndSet3DArray Portion: mismatch at index: Arr(%d,%d,%d)"
                   " computed:%02x, memsetval:%02x\n", z, y, x,
                   array3D[z][y][x], memsetval4seeked);
            break;
          }
        } else {
           if (array3D[z][y][x] != memsetval) {
            testPassed = false;
            printf("seekAndSet3DArray Portion: mismatch at index: Arr(%d,%d,%d)"
                   " computed:%02x, memsetval:%02x\n", z, y, x,
                   array3D[z][y][x], memsetval);
            break;
           }
        }
      }
    }
  }

  HIPCHECK(hipFree(devicePitchedPointer.ptr));
  return testPassed;
}


int main(int argc, char *argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;

  if (p_tests == 1) {
    hipExtent testExtent;
    size_t numH = NUMH_EXT, numW = NUMW_EXT, depth = DEPTH_EXT;

    // Memset with extent width(0) and verify data to be intact
    testExtent = make_hipExtent(0, numH, depth);
    TestPassed &= testMemsetWithExtent(0, testExtent);
    TestPassed &= testMemsetWithExtent(1, testExtent);

    // Memset with extent height(0) and verify data to be intact
    testExtent = make_hipExtent(numW, 0, depth);
    TestPassed &= testMemsetWithExtent(0, testExtent);
    TestPassed &= testMemsetWithExtent(1, testExtent);

    // Memset with extent depth(0) and verify data to be intact
    testExtent = make_hipExtent(numW, numH, 0);
    TestPassed &= testMemsetWithExtent(0, testExtent);
    TestPassed &= testMemsetWithExtent(1, testExtent);

    // Memset with extent width,height,depth as 0 and verify data to be intact
    testExtent = make_hipExtent(0, 0, 0);
    TestPassed &= testMemsetWithExtent(0, testExtent);
    TestPassed &= testMemsetWithExtent(1, testExtent);
  } else if (p_tests == 2) {
    // Memset with max unsigned char and verify memset is success
    TestPassed &= testMemsetMaxValue(0);
    TestPassed &= testMemsetMaxValue(1);
  } else if (p_tests == 3) {
    // Seek and set random slice of 3d array
    TestPassed &= seekAndSet3DArraySlice(0);
    TestPassed &= seekAndSet3DArraySlice(1);
  } else if (p_tests == 4) {
    // Memset selected portion of 3d array
    TestPassed &= seekAndSet3DArrayPortion(0);
    TestPassed &= seekAndSet3DArrayPortion(1);
  } else {
    printf("Didnt receive any valid option. Try options 1 to 4\n");
    TestPassed = false;
  }

  if (TestPassed) {
    passed();
  } else {
    failed("hipMemset3DFunctional validation Failed!");
  }
}
