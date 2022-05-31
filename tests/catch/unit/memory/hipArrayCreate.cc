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
hipArrayCreate API test scenarios
1. Negative Scenarios
2. Allocating Small and big chunk data
3. Multithreaded scenario
*/

#include <hip_test_common.hh>

static constexpr auto NUM_W{4};
static constexpr auto BIGNUM_W{100};
static constexpr auto NUM_H{4};
static constexpr auto BIGNUM_H{100};
static constexpr auto ARRAY_LOOP{100};

/*
 * This API verifies  memory allocations for small and
 * bigger chunks of data.
 * Two scenarios are verified in this API
 * 1. SmallArray: Allocates NUM_W*NUM_H in a loop and
 *    releases the memory and verifies the meminfo.
 * 2. BigArray: Allocates BIGNUM_W*BIGNUM_H in a loop and
 *    releases the memory and verifies the meminfo
 *
 * In both cases, the memory info before allocation and
 * after releasing the memory should be the same.
 *
 */

static void ArrayCreate_DiffSizes(int gpu) {
  HIP_CHECK(hipSetDevice(gpu));
  std::vector<size_t> array_size;
  array_size.push_back(NUM_W);
  array_size.push_back(BIGNUM_W);
  for (auto &size : array_size) {
    HIP_ARRAY array[ARRAY_LOOP];
    size_t tot, avail, ptot, pavail;
    HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
    for (int i = 0; i < ARRAY_LOOP; i++) {
      HIP_ARRAY_DESCRIPTOR desc;
      desc.NumChannels = 1;
      if (size == NUM_W) {
        desc.Width = NUM_W;
        desc.Height = NUM_H;
      } else {
        desc.Width = BIGNUM_W;
        desc.Height = BIGNUM_H;
      }
      desc.Format = HIP_AD_FORMAT_FLOAT;
      HIP_CHECK(hipArrayCreate(&array[i], &desc));
    }
    for (int i = 0; i < ARRAY_LOOP; i++) {
      ARRAY_DESTROY(array[i]);
    }
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    if ((pavail != avail)) {
      HIPASSERT(false);
    }
  }
}

/*Thread function*/
static void ArrayCreateThreadFunc(int gpu) {
  ArrayCreate_DiffSizes(gpu);
}

/* This testcase verifies hipArrayCreate API for small and big chunks data*/
TEST_CASE("Unit_hipArrayCreate_DiffSizes") {
  ArrayCreate_DiffSizes(0);
}


/* This testcase verifies the negative scenarios of
 * hipArrayCreate API
 */
TEST_CASE("Unit_hipArrayCreate_Negative") {
  HIP_ARRAY_DESCRIPTOR desc;
  HIP_ARRAY array;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  desc.NumChannels = 1;
  desc.Width = NUM_W;
  desc.Height = NUM_H;
#if HT_NVIDIA
  SECTION("NullPointer to Array") {
    REQUIRE(hipArrayCreate(nullptr, &desc) != hipSuccess);
  }

  SECTION("NullPointer to Channel Descriptor") {
    REQUIRE(hipArrayCreate(&array, nullptr) != hipSuccess);
  }
#endif
  SECTION("Width 0 for Array Descriptor") {
    desc.Width = 0;
    REQUIRE(hipArrayCreate(&array, &desc) != hipSuccess);
  }

  SECTION("Invalid NumChannels") {
    desc.NumChannels = 3;
    REQUIRE(hipArrayCreate(&array, &desc) != hipSuccess);
  }
}
/*
This testcase verifies the hipArrayCreate API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipArrayCreate API with small and big chunks data
*/
TEST_CASE("Unit_hipArrayCreate_MultiThread") {
  std::vector<std::thread> threadlist;
  int devCnt = 0;

  devCnt = HipTest::getDeviceCount();

  size_t tot, avail, ptot, pavail;
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(ArrayCreateThreadFunc, i));
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

