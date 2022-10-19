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
hipMalloc3D API test scenarios
1. Basic Functionality
2. Negative Scenarios
3. Allocating Small and big chunk data
4. Multithreaded scenario
*/

#include <hip_test_common.hh>
static constexpr auto SMALL_SIZE{4};
static constexpr auto CHUNK_LOOP{100};
static constexpr auto BIG_SIZE{100};
/*
This API verifies hipMalloc3D API by allocating memory in smaller chunks for
CHUNK_LOOP iterations and checks for the memory leaks by get the memory
info before and after the hipMalloc3D API and the difference should
match with the allocated memory
*/
static void MemoryAlloc3DDiffSizes(int gpu) {
  HIPCHECK(hipSetDevice(gpu));
  std::vector<size_t> array_size;
  array_size.push_back(SMALL_SIZE);
  array_size.push_back(BIG_SIZE);
  for (auto &sizes : array_size) {
    size_t width = sizes * sizeof(float);
    size_t height{sizes}, depth{sizes};
    hipPitchedPtr devPitchedPtr[CHUNK_LOOP];
    hipExtent extent = make_hipExtent(width, height, depth);
    size_t tot, avail, ptot, pavail;
    HIPCHECK(hipMemGetInfo(&pavail, &ptot));
    for (int i = 0; i < CHUNK_LOOP; i++) {
      HIPCHECK(hipMalloc3D(&devPitchedPtr[i], extent));
    }
    for (int i = 0; i < CHUNK_LOOP; i++) {
      HIPCHECK(hipFree(devPitchedPtr[i].ptr));
    }
    HIPCHECK(hipMemGetInfo(&avail, &tot));
    if ((pavail != avail)) {
      HIPASSERT(false);
    }
  }
}

static void Malloc3DThreadFunc(int gpu) {
  MemoryAlloc3DDiffSizes(gpu);
}

/*
 * This verifies the hipMalloc3D API by
 * assigning width,height and depth as 10
 */
TEST_CASE("Unit_hipMalloc3D_Basic") {
  size_t width = SMALL_SIZE * sizeof(char);
  size_t height{SMALL_SIZE}, depth{SMALL_SIZE};
  hipPitchedPtr devPitchedPtr;
  hipExtent extent = make_hipExtent(width, height, depth);
  size_t tot, avail, ptot, pavail;
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));

  REQUIRE(hipMalloc3D(&devPitchedPtr, extent) == hipSuccess);
  HIPCHECK(hipFree(devPitchedPtr.ptr));

  HIP_CHECK(hipMemGetInfo(&avail, &tot));

  if (pavail != avail) {
    WARN("Memory leak of hipMalloc3D API in multithreaded scenario");
    REQUIRE(false);
  }
}

/*
This testcase verifies the hipMalloc3D API by allocating
smaller and big chunk data.
*/
TEST_CASE("Unit_hipMalloc3D_SmallandBigChunks") {
  MemoryAlloc3DDiffSizes(0);
}

/*
This testcase verifies the hipMalloc3D API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipMalloc3D API with small and big chunks data
*/
TEST_CASE("Unit_hipMalloc3D_MultiThread") {
  std::vector<std::thread> threadlist;
  int devCnt = 0;

  devCnt = HipTest::getDeviceCount();

  size_t tot, avail, ptot, pavail;
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(Malloc3DThreadFunc, i));
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
