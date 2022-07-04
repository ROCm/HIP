/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of intge, to any person obtaining a copy
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

#include <hip_test_common.hh>
#include <memory>
#include "MemUtils.hh"

/*
 * Description goes in here
 */

using namespace mem_utils;

// value used for memset operations
constexpr int testValue = 0x11;

static void checkForHostSync(hipStream_t stream, bool async, allocType type) {
  if (async && type == allocType::deviceMalloc) {
    HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  } else {
    REQUIRE(true);
  }
  // if (async && type == allocType::deviceMalloc) {
  //   HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  // } else if (!stream && type == allocType::deviceMalloc && !async) {
  //   HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  // } else {
  //   HIP_CHECK(hipStreamQuery(stream));
  // }
}

static void checkForDeviceSync(hipStream_t stream) { HIP_CHECK(hipStreamQuery(stream)); }

// Helper function to run tests for hipMemset allocation types
template <typename T>
static void runMemcpyTests(hipStream_t stream, bool async, allocType type, memType memType,
                           MultiDData data) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  // bool fromHost = GENERATE(true, false);
  bool fromHost = true;

  size_t dataH = data.height == 0 ? 1 : data.height;
  size_t dataD = data.depth == 0 ? 1 : data.depth;

  std::pair<T*, T*> aPtr = initMemory<T>(type, memType, data);
  size_t sizeInBytes = data.width * dataH * dataD;
  createFillerData<T>(sizeInBytes, testValue);
  CAPTURE(type, memType, data.width, data.height, data.depth, stream, async, fromHost, sizeInBytes);

  auto t1 = high_resolution_clock::now();
  launchLongRunningKernel(1000, stream);
  if (fromHost) {
    hostMemcpyCheck(aPtr.first, type, testValue, memType, data, stream, async);
    auto t2 = high_resolution_clock::now();
    printf("Time for a longRunningKernel with : \t%lu ms \n", duration_cast<milliseconds>(t2 - t1));
    printf(
        " Type: \t\t %lu \n Async: \t %lu \n Data W,H,D: \t[ %lu, %lu, %lu ]\n "
        "Stream: \t %lu \n",
        type, async, data.width, data.height, data.depth, stream);

    checkForHostSync(stream, async, type);

  } else {
    deviceMemcpyCheck(aPtr.first, type, testValue, memType, data, stream, async);
    checkForDeviceSync(stream);
  }
  // verify
  HIP_CHECK(hipStreamSynchronize(stream));
  verifyData(aPtr.first, testValue, data, type, memType);
  if (type == allocType::devRegistered) {
    freeStuff(aPtr.second, type);
  } else {
    freeStuff(aPtr.first, type);
  }
}

TEST_CASE("Unit_hipMemcpySync") {
  allocType type = GENERATE(allocType::deviceMalloc, allocType::hostMalloc, allocType::hostRegisted,
                            allocType::devRegistered);
  memType memcpy_type = memType::hipMem;
  MultiDData data;
  // data.width = GENERATE(1, 1024);
  data.width = GENERATE(1024);
  doMemTest<char>(runMemcpyTests<char>, type, memcpy_type, data);
}

TEST_CASE("Unit_hipMemcpy2DSync") {
  //allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
  //                                allocType::hostRegisted, allocType::devRegistered);
  allocType mallocType = allocType::deviceMalloc;
  // allocType mallocType = allocType::deviceMalloc;
  memType memcpy_type = memType::hipMem2D;
  MultiDData data;
  data.width = GENERATE(1, 1024);
  data.height = GENERATE(1, 1024);
  

  doMemTest<char>(runMemcpyTests<char>, mallocType, memcpy_type, data);
}

TEST_CASE("Unit_hipMemcpy3DSync") {
  // allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
  //                                 allocType::hostRegisted, allocType::devRegistered);
  allocType mallocType = allocType::deviceMalloc;
  memType memcpy_type = memType::hipMem3D;
  MultiDData data;
  //data.width = GENERATE(1, 256);
  //data.height = GENERATE(1, 256);
  //data.depth = GENERATE(1, 256);
   data.width = 128;
   data.height = 128;
   data.depth = 129;


  doMemTest<char>(runMemcpyTests<char>, mallocType, memcpy_type, data);
}