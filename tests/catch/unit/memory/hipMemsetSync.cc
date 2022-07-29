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

#include "MemUtils.hh"

/*
 * These testcases verify that synchronous memset functions are asynchronous with respect to the
 * host except when the target is pinned host memory or a Unified Memory region
 */

using namespace memset_utils;

// value used for memset operations
constexpr int testValue = 0x11;

// Helper function to run tests for hipMemset allocation types
template <typename T>
void runTests(hipStream_t stream, bool async, allocType type, memSetType memsetType,
              MultiDData data) {
  CAPTURE(type, memsetType, data.width, data.height, data.depth, stream, async);
  std::pair<T*, T*> aPtr = initMemory<T>(type, memsetType, data);
  runKernelForMs(100, stream);
  memsetCheck(aPtr.first, testValue, memsetType, data, stream, async);

  if (async || type == allocType::deviceMalloc) {
    HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  } else {
    HIP_CHECK(hipStreamQuery(stream));
  }

  HIP_CHECK(hipStreamSynchronize(stream));
  verifyData(aPtr.first, testValue, data, type, memsetType);

  if (type == allocType::devRegistered) {
    freeStuff(aPtr.second, type);
  } else {
    freeStuff(aPtr.first, type);
  }
}

TEST_CASE("Unit_hipMemsetSync") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-86");
  return;
#endif
  allocType type = GENERATE(allocType::deviceMalloc, allocType::hostMalloc, allocType::hostRegisted,
                            allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset;
  MultiDData data;
  data.width = GENERATE(1, 1024);
  doMemsetTest<char>(runTests<char>, type, memset_type, data);
}

TEMPLATE_TEST_CASE("Unit_hipMemsetDSync", "", int8_t, int16_t, uint32_t) {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-86");
  return;
#endif
  allocType mallocType = GENERATE(allocType::hostRegisted, allocType::deviceMalloc,
                                  allocType::hostMalloc, allocType::devRegistered);
  memSetType memset_type;
  MultiDData data;
  data.width = GENERATE(1, 1024);

  if (std::is_same<int8_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD8;
  } else if (std::is_same<int16_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD16;
  } else if (std::is_same<uint32_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD32;
  }

  doMemsetTest<TestType>(runTests<char>, mallocType, memset_type, data);
}

TEST_CASE("Unit_hipMemset2DSync") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-86");
  return;
#endif
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset2D;
  MultiDData data;
  data.width = GENERATE(1, 1024);
  data.height = GENERATE(1, 1024);

  doMemsetTest<char>(runTests<char>, mallocType, memset_type, data);
}

TEST_CASE("Unit_hipMemset3DSync") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-86");
  return;
#endif
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset3D;
  MultiDData data;
  data.width = GENERATE(1, 256);
  data.height = GENERATE(1, 256);
  data.depth = GENERATE(1, 256);

  doMemsetTest<char>(runTests<char>, mallocType, memset_type, data);
}
