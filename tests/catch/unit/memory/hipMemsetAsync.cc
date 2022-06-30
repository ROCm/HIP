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
 * This testcase verifies that asynchronous memset functions are asynchronous with respect to the
 * host except when the target is pinned host memory or a Unified Memory region
 */

constexpr int testValue1 = 97;
constexpr int testValue2 = 98;


using namespace memset_utils;

// Helper function to run tests for hipMemset allocation types
template <typename T>
void runAsyncTests(hipStream_t stream, allocType type, memSetType memsetType, MultiDData data1,
                   MultiDData data2) {
  std::pair<T*, T*> aPtr{};
  MultiDData totalRange;
  totalRange.width = data1.width + data2.width;
  totalRange.height = data1.height + data2.height;
  totalRange.depth = data1.depth + data2.depth;
  aPtr = initMemory<T>(type, memsetType, totalRange);
  data1.pitch = totalRange.pitch;
  data2.pitch = totalRange.pitch;

  memsetCheck(aPtr.first, testValue1, memsetType, data1, stream);
  memsetCheck(aPtr.first, testValue2, memsetType, data2, stream);

  HIP_CHECK(hipStreamSynchronize(stream));
  verifyData(aPtr.first, testValue1, data1, type, memsetType);
  verifyData(aPtr.first, testValue2, data2, type, memsetType);


  if (type == allocType::devRegistered) {
    freeStuff(aPtr.second, type);
  } else {
    freeStuff(aPtr.first, type);
  }
}

/*
 * test 2 async hipMemset's on the same memory at different offsets
 */

TEST_CASE("Unit_hipMemsetASyncMulti") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-127");
  return;
#endif
  allocType mallocType = GENERATE(allocType::hostMalloc, allocType::deviceMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  MultiDData data2;
  data2.width = data1.width;

  data2.offset = data1.width;
  doMemsetTest<char>(runAsyncTests<char>, mallocType, memset_type, data1, data2);
}

/*
 * test 2 async hipMemsetD[8,16,32]'s on the same memory at different offsets
 */
TEMPLATE_TEST_CASE("Unit_hipMemsetDASyncMulti", "", int8_t, int16_t, uint32_t) {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-127");
  return;
#endif
  allocType mallocType = GENERATE(allocType::hostRegisted, allocType::deviceMalloc,
                                  allocType::hostMalloc, allocType::devRegistered);
  memSetType memset_type;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  MultiDData data2;
  data2.width = data1.width;
  data2.offset = data1.width;

  if (std::is_same<int8_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD8;
  } else if (std::is_same<int16_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD16;
  } else if (std::is_same<uint32_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD32;
  }
  doMemsetTest<TestType>(runAsyncTests<TestType>, mallocType, memset_type, data1, data2);
}
/*
 * test 2 async hipMemset2D's on the same memory at different offsets
 */
TEST_CASE("Unit_hipMemset2DASyncMulti") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-127");
  return;
#endif
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset2D;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  data1.height = data1.width;
  MultiDData data2;
  data2.width = data1.width;
  data2.height = data1.height;
  data2.offset = data1.width;

  doMemsetTest<char>(runAsyncTests<char>, mallocType, memset_type, data1, data2);
}
/*
 * test 2 async hipMemset3D's on the same memory at different offsets
 */
TEST_CASE("Unit_hipMemset3DASyncMulti") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-127");
  return;
#endif
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset3D;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 256);
  data1.height = data1.width;
  data1.depth = data1.width;
  MultiDData data2;
  data2.width = data1.width;
  data2.height = data1.width;
  data2.depth = data1.width;
  data2.offset = data1.width;

  doMemsetTest<char>(runAsyncTests<char>, mallocType, memset_type, data1, data2);
}
