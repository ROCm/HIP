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

/*
hipMalloc3DArray API test scenarios
1. Basic Functionality
2. Negative Scenarios
3. Allocating Small and big chunk data
4. Multithreaded scenario
*/



#include <hip_test_common.hh>

static constexpr auto ARRAY_SIZE{4};
static constexpr auto BIG_ARRAY_SIZE{100};
static constexpr auto ARRAY_LOOP{100};

/*
 * This API verifies  memory allocations for small and
 * bigger chunks of data.
 * Two scenarios are verified in this API
 * 1. SmallArray: Allocates ARRAY_SIZE in a loop and
 *    releases the memory and verifies the meminfo.
 * 2. BigArray: Allocates BIG_ARRAY_SIZE in a loop and
 *    releases the memory and verifies the meminfo
 *
 * In both cases, the memory info before allocation and
 * after releasing the memory should be the same
 *
 */
static void Malloc3DArray_DiffSizes(int gpu) {
  HIP_CHECK(hipSetDevice(gpu));
  std::vector<int> array_size;
  array_size.push_back(ARRAY_SIZE);
  array_size.push_back(BIG_ARRAY_SIZE);
  for (auto &size : array_size) {
    int width{size}, height{size}, depth{size};
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(float)*8, 0,
        0, 0, hipChannelFormatKindFloat);
    hipArray *arr[ARRAY_LOOP];
    size_t tot, avail, ptot, pavail;
    HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
    for (int i = 0; i < ARRAY_LOOP; i++) {
      HIP_CHECK(hipMalloc3DArray(&arr[i], &channelDesc, make_hipExtent(width,
              height, depth), hipArrayDefault));
    }
    for (int i = 0; i < ARRAY_LOOP; i++) {
      HIP_CHECK(hipFreeArray(arr[i]));
    }
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    if ((pavail != avail)) {
      HIPASSERT(false);
    }
  }
}

/* Thread Function */
static void Malloc3DArrayThreadFunc(int gpu) {
  Malloc3DArray_DiffSizes(gpu);
}

/*
 * Verifies the negative scenarios of hipMalloc3DArray API
 */
TEST_CASE("Unit_hipMalloc3DArray_Negative") {
  constexpr int width{ARRAY_SIZE}, height{ARRAY_SIZE}, depth{ARRAY_SIZE};
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(float)*8, 0,
                                     0, 0, hipChannelFormatKindFloat);
  hipArray *arr;
#if HT_NVIDIA
  SECTION("NullPointer to Array") {
    REQUIRE(hipMalloc3DArray(nullptr, &channelDesc, make_hipExtent(width,
            height, depth), hipArrayDefault) != hipSuccess);
  }

  SECTION("NullPointer to Channel Descriptor") {
    REQUIRE(hipMalloc3DArray(&arr, nullptr, make_hipExtent(width,
            height, depth), hipArrayDefault) != hipSuccess);
  }
#endif
  SECTION("Width 0 in hipExtent") {
    REQUIRE(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(0,
            height, width), hipArrayDefault) != hipSuccess);
  }

  SECTION("Height 0 in hipExtent") {
    REQUIRE(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width,
            0, width), hipArrayDefault) != hipSuccess);
  }

  SECTION("Invalid Flag") {
    REQUIRE(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width,
            height, depth), 100) != hipSuccess);
  }

  SECTION("Width,Height & Depth 0 in hipExtent") {
    REQUIRE(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(0,
            0, 0), hipArrayDefault) != hipSuccess);
  }

  SECTION("Max int values to extent") {
    REQUIRE(hipMalloc3DArray(&arr, &channelDesc,
            make_hipExtent(std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max()),
            hipArrayDefault) != hipSuccess);
  }
}
/*
 * Verifies the extent validation scenarios
 * 1. Passing depth as 0 would create 2D array
 * 2. Passing height and depth as 0 would create 1D array
 * from hipMalloc3DArray API
 */
TEST_CASE("Unit_hipMalloc3DArray_ExtentValidation") {
  constexpr int width{ARRAY_SIZE}, height{ARRAY_SIZE};
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(float)*8, 0,
                                     0, 0, hipChannelFormatKindFloat);
  hipArray *arr;

  SECTION("Depth 0 in hipExtent") {
    REQUIRE(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width,
            height, 0), hipArrayDefault) == hipSuccess);
    HIP_CHECK(hipFreeArray(arr));
  }

  SECTION("Height & Depth 0 in hipExtent") {
    REQUIRE(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width,
            0, 0), hipArrayDefault) == hipSuccess);
    HIP_CHECK(hipFreeArray(arr));
  }
}

/*
 * Verifies hipMalloc3DArray API by passing width,height
 * and depth as 10
 */
TEST_CASE("Unit_hipMalloc3DArray_Basic") {
  constexpr int width{ARRAY_SIZE}, height{ARRAY_SIZE}, depth{ARRAY_SIZE};
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(float)*8, 0,
                                     0, 0, hipChannelFormatKindFloat);
  hipArray *arr;

  REQUIRE(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width,
          height, depth), hipArrayDefault) == hipSuccess);
  HIP_CHECK(hipFreeArray(arr));
}

TEST_CASE("Unit_hipMalloc3DArray_DiffSizes") {
  Malloc3DArray_DiffSizes(0);
}
/*
This testcase verifies the hipMalloc3DArray API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipMalloc3DArray API with small and big chunks data
*/
TEST_CASE("Unit_hipMalloc3DArray_MultiThread") {
  std::vector<std::thread> threadlist;
  int devCnt = 0;
  devCnt = HipTest::getDeviceCount();
  size_t tot, avail, ptot, pavail;
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(Malloc3DArrayThreadFunc, i));
  }

  for (auto &t : threadlist) {
    t.join();
  }
  HIP_CHECK(hipMemGetInfo(&avail, &tot));

  if (pavail != avail) {
    WARN("Memory leak of hipMalloc3D API in multithreaded scenario");
    REQUIRE(false);
  }
}

