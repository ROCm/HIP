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

 9) Validate data after performing memory set operation with max memset value
 for hipMemset3D api.
 10) Validate data after performing memory set operation with max memset value
 for hipMemset3DAsync api.

 11) Select random slice of 3d array and Memset complete slice with
 hipMemset3D api.
 12) Select random slice of 3d array and Memset complete slice with
 hipMemset3DAsync api.

 13) Seek device pitched ptr to desired portion of 3d array and memset the
 portion with hipMemset3D api.
 14) Seek device pitched ptr to desired portion of 3d array and memset the
 portion with hipMemset3DAsync api.
*/

#include <hip_test_common.hh>

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
static void testMemsetWithExtent(bool bAsync, hipExtent tstExtent) {
  hipPitchedPtr devPitchedPtr;
  hipError_t ret;
  char *A_h;
  size_t numH = NUMH_EXT, numW = NUMW_EXT, depth = DEPTH_EXT;
  size_t width = numW * sizeof(char);
  hipExtent extent = make_hipExtent(width, numH, depth);

  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;

  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);
  memset(A_h, 0, sizeElements);
  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  if (bAsync) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    ret = hipMemset3DAsync(devPitchedPtr, MEMSETVAL, extent, stream);
    INFO("testMemsetWithExtent(" << extent.width << "," << extent.height
                                 << "," << extent.depth << ") memset "
                                 << MEMSETVAL << ", ret : " << ret);
    REQUIRE(ret == hipSuccess);

    ret = hipMemset3DAsync(devPitchedPtr, TESTVAL, tstExtent, stream);
    INFO("testMemsetWithExtent(" << tstExtent.width << "," << tstExtent.height
                                 << "," << tstExtent.depth << ") memset "
                                 << TESTVAL << "ret : " << ret);
    REQUIRE(ret == hipSuccess);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  } else {
    ret = hipMemset3D(devPitchedPtr, MEMSETVAL, extent);
    INFO("testMemsetWithExtent(" << extent.width << "," << extent.height
                                 << "," << extent.depth << ") memset "
                                 << MEMSETVAL << ",ret : " << ret);
    REQUIRE(ret == hipSuccess);

    ret = hipMemset3D(devPitchedPtr, TESTVAL, tstExtent);
    INFO("testMemsetWithExtent(" << tstExtent.width << "," << tstExtent.height
                                 << "," << tstExtent.depth << ") memset "
                                 << TESTVAL << ",ret : " << ret);
    REQUIRE(ret == hipSuccess);
  }


  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIP_CHECK(hipMemcpy3D(&myparms));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != MEMSETVAL) {
      INFO("testMemsetWithExtent: index:" << i << ",computed:"
               << std::hex << static_cast<int>(A_h[i]) << ",memsetval:"
                                            << std::hex << MEMSETVAL);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}


/**
 * Validates data after performing memory set operation with max memset value
 */
static void testMemsetMaxValue(bool bAsync) {
  hipPitchedPtr devPitchedPtr;
  unsigned char *A_h;
  int memsetval = std::numeric_limits<unsigned char>::max();
  size_t numH = NUMH_MAX, numW = NUMW_MAX, depth = DEPTH_MAX;
  size_t width = numW * sizeof(unsigned char);
  hipExtent extent = make_hipExtent(width, numH, depth);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW* numH* depth;

  A_h = reinterpret_cast<unsigned char *> (malloc(sizeElements));
  REQUIRE(A_h != nullptr);
  memset(A_h, 0, sizeElements);

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  if (bAsync) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  } else {
    HIP_CHECK(hipMemset3D(devPitchedPtr, memsetval, extent));
  }

  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIP_CHECK(hipMemcpy3D(&myparms));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("testMemsetMaxValue: index:" << i << ",computed:"
               << std::hex << static_cast<int>(A_h[i]) << ",memsetval:"
                                            << std::hex << memsetval);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}

/**
 * Function seeks device ptr to random slice and performs Memset operation
 * on the slice selected.
 */
static void seekAndSet3DArraySlice(bool bAsync) {
  char array3D[ZSIZE_S][YSIZE_S][XSIZE_S]{};
  dim3 arr_dimensions = dim3(ZSIZE_S, YSIZE_S, XSIZE_S);
  hipExtent extent = make_hipExtent(sizeof(char) * arr_dimensions.x,
                                    arr_dimensions.y, arr_dimensions.z);
  hipPitchedPtr devicePitchedPointer;
  int memsetval = MEMSETVAL, memsetval4seeked = TESTVAL;

  HIP_CHECK(hipMalloc3D(&devicePitchedPointer, extent));
  HIP_CHECK(hipMemset3D(devicePitchedPointer, memsetval, extent));

  // select random slice for memset
  unsigned int seed = time(nullptr);
  int slice_index = HipTest::RAND_R(&seed) % ZSIZE_S;

  INFO("memset3d for sliceindex " << slice_index);

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
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemset3DAsync(modDevPitchedPtr, memsetval4seeked,
                              extentSlice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  } else {
    // Memset selected slice
    HIP_CHECK(hipMemset3D(modDevPitchedPtr, memsetval4seeked, extentSlice));
  }

  // Copy result back to host buffer
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(array3D, sizeof(char) * arr_dimensions.x,
                                      arr_dimensions.x, arr_dimensions.y);
  myparms.srcPtr = devicePitchedPointer;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIP_CHECK(hipMemcpy3D(&myparms));

  for (int z = 0; z < ZSIZE_S; z++) {
    for (int y = 0; y < YSIZE_S; y++) {
      for (int x = 0; x < XSIZE_S; x++) {
        if (z == slice_index) {
          if (array3D[z][y][x] != memsetval4seeked) {
            INFO("seekAndSet3DArray Slice: mismatch at index: Arr(" << z
                   << "," << y << "," << x << ") " << "computed:" << std::hex
                   << array3D[z][y][x] << ", memsetval:" << std::hex
                   << memsetval4seeked);
            REQUIRE(false);
          }
        } else {
          if (array3D[z][y][x] != memsetval) {
            INFO("seekAndSet3DArray Slice: mismatch at index: Arr(" << z
                   << "," << y << "," << x << ") " << "computed:" << std::hex
                   << array3D[z][y][x] << ", memsetval:" << std::hex
                   << memsetval);
            REQUIRE(false);
          }
        }
      }
    }
  }

  HIP_CHECK(hipFree(devicePitchedPointer.ptr));
}

/**
 * Function seeks device ptr to selected portion of 3d array
 * and performs Memset operation on the portion.
 */
static void seekAndSet3DArrayPortion(bool bAsync) {
  char array3D[ZSIZE_P][YSIZE_P][XSIZE_P]{};
  dim3 arr_dimensions = dim3(ZSIZE_P, YSIZE_P, XSIZE_P);
  hipExtent extent = make_hipExtent(sizeof(char) * arr_dimensions.x,
                                    arr_dimensions.y, arr_dimensions.z);
  hipPitchedPtr devicePitchedPointer;
  int memsetval = MEMSETVAL, memsetval4seeked = TESTVAL;

  HIP_CHECK(hipMalloc3D(&devicePitchedPointer, extent));
  HIP_CHECK(hipMemset3D(devicePitchedPointer, memsetval, extent));

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
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemset3DAsync(modDevPitchedPtr, memsetval4seeked,
                              setExtent, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  } else {
    // Memset selected portion
    HIP_CHECK(hipMemset3D(modDevPitchedPtr, memsetval4seeked, setExtent));
  }

  // Copy result back to host buffer
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(array3D, sizeof(char) * arr_dimensions.x,
                                      arr_dimensions.x, arr_dimensions.y);
  myparms.srcPtr = devicePitchedPointer;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif

  HIP_CHECK(hipMemcpy3D(&myparms));

  for (int z = 0; z < ZSIZE_P; z++) {
    for (int y = 0; y < YSIZE_P; y++) {
      for (int x = 0; x < XSIZE_P; x++) {
        if ((z >= ZPOS_START && z <= ZPOS_END) &&
            (y >= YPOS_START && y <= YPOS_END) &&
            (x >= XPOS_START && x <= XPOS_END)) {
          if (array3D[z][y][x] != memsetval4seeked) {
            INFO("seekAndSet3DArray Portion: mismatch at index: Arr(" << z
                   << "," << y << "," << x << ") " << "computed:" << std::hex
                   << array3D[z][y][x] << ", memsetval:" << std::hex
                   << memsetval4seeked);
            REQUIRE(false);
          }
        } else {
           if (array3D[z][y][x] != memsetval) {
            INFO("seekAndSet3DArray Portion: mismatch at index: Arr(" << z
                   << "," << y << "," << x << ") " << "computed:" << std::hex
                   << array3D[z][y][x] << ", memsetval:" << std::hex
                   << memsetval);
            REQUIRE(false);
           }
        }
      }
    }
  }

  HIP_CHECK(hipFree(devicePitchedPointer.ptr));
}



/**
 * Test Memset3D with different combinations of extent
 * taking zero and non-zero fields.
 */
TEST_CASE("Unit_hipMemset3D_MemsetWithExtent") {
  hipExtent testExtent;
  size_t numH = NUMH_EXT, numW = NUMW_EXT, depth = DEPTH_EXT;

  SECTION("Memset with extent width(0)") {
    // Memset with extent width(0) and verify data to be intact
    testExtent = make_hipExtent(0, numH, depth);
    testMemsetWithExtent(0, testExtent);
  }

  SECTION("Memset with extent height(0)") {
    // Memset with extent height(0) and verify data to be intact
    testExtent = make_hipExtent(numW, 0, depth);
    testMemsetWithExtent(0, testExtent);
  }

  SECTION("Memset with extent depth(0)") {
    // Memset with extent depth(0) and verify data to be intact
    testExtent = make_hipExtent(numW, numH, 0);
    testMemsetWithExtent(0, testExtent);
  }

  SECTION("Memset with extent width,height,depth as 0") {
    // Memset with extent width,height,depth as 0 and verify data to be intact
    testExtent = make_hipExtent(0, 0, 0);
    testMemsetWithExtent(0, testExtent);
  }
}


/**
 * Test Memset3DAsync with different combinations of extent
 * taking zero and non-zero fields.
 */
TEST_CASE("Unit_hipMemset3DAsync_MemsetWithExtent") {
  hipExtent testExtent;
  size_t numH = NUMH_EXT, numW = NUMW_EXT, depth = DEPTH_EXT;

  SECTION("Memset with extent width(0)") {
    // Memset with extent width(0) and verify data to be intact
    testExtent = make_hipExtent(0, numH, depth);
    testMemsetWithExtent(1, testExtent);
  }

  SECTION("Memset with extent height(0)") {
    // Memset with extent height(0) and verify data to be intact
    testExtent = make_hipExtent(numW, 0, depth);
    testMemsetWithExtent(1, testExtent);
  }

  SECTION("Memset with extent depth(0)") {
    // Memset with extent depth(0) and verify data to be intact
    testExtent = make_hipExtent(numW, numH, 0);
    testMemsetWithExtent(1, testExtent);
  }

  SECTION("Memset with extent width,height,depth as 0") {
    // Memset with extent width,height,depth as 0 and verify data to be intact
    testExtent = make_hipExtent(0, 0, 0);
    testMemsetWithExtent(1, testExtent);
  }
}

/**
 * Memset3D with max unsigned char and verify memset operation is success
 */
TEST_CASE("Unit_hipMemset3D_MemsetMaxValue") {
  testMemsetMaxValue(0);
}

/**
 * Memset3DAsync with max unsigned char and verify memset operation is success
 */
TEST_CASE("Unit_hipMemset3DAsync_MemsetMaxValue") {
  testMemsetMaxValue(1);
}

/**
 * Seek and set random slice of 3d array, verify memset is success
 */
TEST_CASE("Unit_hipMemset3D_SeekSetSlice") {
  seekAndSet3DArraySlice(0);
}

/**
 * Seek and set random slice of 3d array with async, verify memset is success
 */
TEST_CASE("Unit_hipMemset3DAsync_SeekSetSlice") {
  seekAndSet3DArraySlice(1);
}

/**
 * Memset3D selected portion of 3d array
 */
TEST_CASE("Unit_hipMemset3D_SeekSetArrayPortion") {
  seekAndSet3DArrayPortion(0);
}

/**
 * Memset3DAsync selected portion of 3d array
 */
TEST_CASE("Unit_hipMemset3DAsync_SeekSetArrayPortion") {
  seekAndSet3DArrayPortion(1);
}
