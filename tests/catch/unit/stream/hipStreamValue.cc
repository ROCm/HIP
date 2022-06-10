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
#include <hip_test_common.hh>

constexpr unsigned int writeFlag = 0;

#define DEFINE_HIP_STREAM_VALUE(TYPE, BITS, ...) hipStream##TYPE##Value##BITS(__VA_ARGS__)

#define CHECK_HIP_STREAM_VALUE(TYPE, BITS, ...)                                                    \
  HIP_CHECK(DEFINE_HIP_STREAM_VALUE(TYPE, BITS, __VA_ARGS__));

#define NEG_TEST_ERROR_CHECK(TYPE, BITS, errorCode, ...)                                           \
  HIP_CHECK_ERROR(DEFINE_HIP_STREAM_VALUE(TYPE, BITS, __VA_ARGS__), errorCode);

#if HT_AMD
// Random predefiend 32 and 64 bit values
constexpr uint32_t value32 = 0x70F0F0FF;
constexpr uint64_t value64 = 0x7FFF0000FFFF0000;
constexpr uint32_t DATA_INIT = 0x1234;
constexpr uint32_t DATA_UPDATE = 0X4321;

template <typename intT> struct TEST_WAIT {
  using uintT = typename std::make_unsigned<intT>::type;
  int compareOp;
  uintT mask;
  uintT waitValue;
  intT signalValueFail;
  intT signalValuePass;

  TEST_WAIT(int compareOp, uintT waitValue, intT signalValueFail, intT signalValuePass)
      : compareOp{compareOp},
        waitValue{waitValue},
        signalValueFail{signalValueFail},
        signalValuePass{signalValuePass} {
    mask = static_cast<uintT>(0xFFFFFFFFFFFFFFFF);
  }

  TEST_WAIT(int compareOp, uintT mask, uintT waitValue, intT signalValueFail, intT signalValuePass)
      : compareOp{compareOp},
        mask{mask},
        waitValue{waitValue},
        signalValueFail{signalValueFail},
        signalValuePass{signalValuePass} {}
};
typedef TEST_WAIT<int32_t> TEST_WAIT32;
typedef TEST_WAIT<int64_t> TEST_WAIT64;

bool streamWaitValueSupported() {
  int device_num = 0;
  HIP_CHECK(hipGetDeviceCount(&device_num));
  int waitValueSupport;
  for (int device_id = 0; device_id < device_num; ++device_id) {
    HIP_CHECK(hipSetDevice(device_id));
    waitValueSupport = 0;
    HIP_CHECK(hipDeviceGetAttribute(&waitValueSupport, hipDeviceAttributeCanUseStreamWaitValue,
                                    device_id));
    if (waitValueSupport == 1) return true;
  }
  return false;
}

// hipStreamWriteValue Tests
TEST_CASE("Unit_hipStreamValue_Write") {
  int64_t* signalPtr;

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate Host Memory
  auto hostPtr64 = std::unique_ptr<uint64_t>(new uint64_t(1));
  auto hostPtr32 = std::unique_ptr<uint32_t>(new uint32_t(1));

  // Register Host Memory
  HIP_CHECK(hipHostRegister(hostPtr64.get(), sizeof(int64_t), 0));
  HIP_CHECK(hipHostRegister(hostPtr32.get(), sizeof(int32_t), 0));

  // Register Signal Memory
  HIP_CHECK(hipExtMallocWithFlags((void**)&signalPtr, 8, hipMallocSignalMemory));

  // Initialise Data
  *signalPtr = 0x0;
  *hostPtr64 = 0x0;
  *hostPtr32 = 0x0;

  SECTION("Registered host memory hipStreamWriteValue32") {
    INFO("Test writting to registered host pointer using hipStreamWriteValue32");
    HIP_CHECK(hipStreamWriteValue32(stream, hostPtr32.get(), value32, writeFlag));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_ASSERT(*hostPtr32 == value32);
  }

  SECTION("Registered host memory hipStreamWriteValue64") {
    INFO("Test writting to registered host pointer using hipStreamWriteValue32");
    HIP_CHECK(hipStreamWriteValue64(stream, hostPtr64.get(), value64, writeFlag));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_ASSERT(*hostPtr64 == value64);
  }

  // Test writting device pointer
  void* devicePtr64;
  void* devicePtr32;
  HIP_CHECK(hipHostGetDevicePointer((void**)&devicePtr64, hostPtr64.get(), 0));
  HIP_CHECK(hipHostGetDevicePointer((void**)&devicePtr32, hostPtr32.get(), 0));
  // Reset values
  *hostPtr64 = 0x0;
  *hostPtr32 = 0x0;

  SECTION("Device Memory hipStreamWriteValue32") {
    INFO("Test writting to device pointer using hipStreamWriteValue32");
    HIP_CHECK(hipStreamWriteValue32(stream, devicePtr32, value32, writeFlag));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_ASSERT(*hostPtr32 == value32);
  }

  SECTION("Device Memory hipStreamWriteValue64") {
    INFO("Test writting to device pointer using hipStreamWriteValue64");
    HIP_CHECK(hipStreamWriteValue64(stream, devicePtr64, value64, writeFlag));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_ASSERT(*hostPtr64 == value64);
  }

  // Test Writing to Signal Memory
  SECTION("Signal Memory hipStreamWriteValue64") {
    INFO("Test writting to signal memory using hipStreamWriteValue64");
    HIP_CHECK(hipStreamWriteValue64(stream, signalPtr, value64, writeFlag));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_ASSERT(*signalPtr == value64);
  }

  // Cleanup
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostUnregister(hostPtr64.get()));
  HIP_CHECK(hipHostUnregister(hostPtr32.get()));
  HIP_CHECK(hipFree(signalPtr));
}

// hipStreamWaitValue Tests
template <bool isBlocking, typename intT, typename TEST_T>
void initData(intT* dataPtr, int64_t* signalPtr, TEST_T tc, std::vector<hipEvent_t>& events) {
  // Initialize memory to be waited on
  *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;


  // Initialize host pointers
  dataPtr[0] = DATA_INIT;
  dataPtr[1] = DATA_INIT;


  hipEvent_t firstWriteEvent{nullptr};
  hipEvent_t secondWriteEvent{nullptr};
  HIP_CHECK(hipEventCreate(&firstWriteEvent));
  HIP_CHECK(hipEventCreate(&secondWriteEvent));
  events.push_back(firstWriteEvent);
  events.push_back(secondWriteEvent);
}

template <bool isBlocking, typename intT, typename TEST_T>
void syncAndCheckData(hipStream_t stream, intT* dataPtr, int64_t* signalPtr, TEST_T tc,
                      std::vector<hipEvent_t>& events) {
  // Ensure first part of host memory is updated
  HIP_CHECK(hipStreamWaitEvent(stream, events[0], 0));
  HIP_ASSERT(dataPtr[0] == DATA_UPDATE);
  if (isBlocking) {
    // Ensure second part of host memory isn't updated yet
    HIP_ASSERT(hipEventQuery(events[1]) == hipErrorNotReady);
    HIP_ASSERT(dataPtr[1] == DATA_INIT);
    // Update value to release stream
    *signalPtr = tc.signalValuePass;
  }

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_ASSERT(hipEventQuery(events[1]) == hipSuccess);
  // Finally ensure that second part of host memory is updated
  HIP_ASSERT(dataPtr[1] == DATA_UPDATE);
}

template <typename intT> void cleanup(hipStream_t& stream, intT* dataPtr, int64_t* signalPtr) {
  // Cleanup
  HIP_CHECK(hipFree(signalPtr));
  HIP_CHECK(hipHostUnregister(dataPtr));
  HIP_CHECK(hipStreamDestroy(stream));
}

template <typename intT, bool isBlocking, typename TEST_T> void testWait(TEST_T tc) {
  if (!streamWaitValueSupported()) {
    UNSCOPED_INFO(" hipStreamWaitValue: not supported on this device , skipping ...");
    return;
  }

  // Initialize stream
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate Host Memory
  std::unique_ptr<intT> dataPtr(new intT(2));

  // Register Host Memory
  HIP_CHECK(hipHostRegister(&(dataPtr.get()[0]), sizeof(intT), 0));
  HIP_CHECK(hipHostRegister(&(dataPtr.get()[1]), sizeof(intT), 0));

  // Allocate Signal Memory
  int64_t* signalPtr;
  HIP_CHECK(hipExtMallocWithFlags((void**)&signalPtr, 8, hipMallocSignalMemory));

  std::vector<hipEvent_t> events;
  initData<isBlocking>(dataPtr.get(), signalPtr, tc, events);

  if (std::is_same<intT, int32_t>::value) {
    CHECK_HIP_STREAM_VALUE(Write, 32, stream, &(dataPtr.get()[0]), DATA_UPDATE, writeFlag)
    HIP_CHECK(hipEventRecord(events[0], stream));

    if (static_cast<uint32_t>(tc.mask) != 0xFFFFFFFF) {
      CHECK_HIP_STREAM_VALUE(Wait, 32, stream, signalPtr, static_cast<uint32_t>(tc.waitValue),
                             tc.compareOp, static_cast<uint32_t>(tc.mask));
    } else {
      CHECK_HIP_STREAM_VALUE(Wait, 32, stream, signalPtr, tc.waitValue, tc.compareOp);
    }

    CHECK_HIP_STREAM_VALUE(Write, 32, stream, &(dataPtr.get()[1]), DATA_UPDATE, writeFlag)
  } else {
    CHECK_HIP_STREAM_VALUE(Write, 64, stream, &(dataPtr.get()[0]), DATA_UPDATE, writeFlag)
    HIP_CHECK(hipEventRecord(events[0], stream));

    if (tc.mask != 0xFFFFFFFFFFFFFFFF) {
      CHECK_HIP_STREAM_VALUE(Wait, 64, stream, signalPtr, tc.waitValue, tc.compareOp, tc.mask);
    } else {
      CHECK_HIP_STREAM_VALUE(Wait, 64, stream, signalPtr, tc.waitValue, tc.compareOp);
    }

    CHECK_HIP_STREAM_VALUE(Write, 64, stream, &(dataPtr.get()[1]), DATA_UPDATE, writeFlag)
  }

  HIP_CHECK(hipEventRecord(events[1], stream));

  syncAndCheckData<isBlocking>(stream, dataPtr.get(), signalPtr, tc, events);
  cleanup(stream, dataPtr.get(), signalPtr);
}
#undef CHECK_HIP_STREAM_VALUE

#define DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32(suffix, test_t)                                    \
  TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_" + std::string(suffix)) {                        \
    testWait<int32_t, true>(test_t);                                                               \
  }                                                                                                \
  TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_" + std::string(suffix)) {                     \
    testWait<int32_t, false>(test_t);                                                              \
  }

// Using Mask
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Gte_1",
                                        TEST_WAIT64(  // mask will ignore few MSB bits
                                            hipStreamWaitValueGte, 0x0000FFFFFFFFFFFF,
                                            0x000000007FFF0001, 0x7FFF00007FFF0000,
                                            0x000000007FFF0001))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Gte_2",
                                        TEST_WAIT64(hipStreamWaitValueGte, 0xF, 0x4, 0x3, 0x6))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Eq_1",
                                        TEST_WAIT64(  // mask will ignore few MSB bits
                                            hipStreamWaitValueEq, 0x0000FFFFFFFFFFFF,
                                            0x000000000FFF0001, 0x7FFF00000FFF0000,
                                            0x7F0000000FFF0001))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Eq_2",
                                        TEST_WAIT64(hipStreamWaitValueEq, 0xFF, 0x11, 0x25, 0x11))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_And",
                                        TEST_WAIT64(  // mask will discard bits 8 to 11
                                            hipStreamWaitValueAnd, 0xFF, 0xF4A, 0xF35, 0X02))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Nor_1",
                                        TEST_WAIT64(  // mask is set to ignore the sign bit.
                                            hipStreamWaitValueNor, 0x7FFFFFFFFFFFFFFF,
                                            0x7FFFFFFFFFFFF247, 0x7FFFFFFFFFFFFdbd,
                                            0x7FFFFFFFFFFFFdb5))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Nor_2",
                                        TEST_WAIT64(  // mask is set to apply NOR for bits 0 to 3.
                                            hipStreamWaitValueNor, 0xF, 0x7E, 0x7D, 0x76))

// Not Using Mask
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("NoMask_Eq",
                                        TEST_WAIT32(hipStreamWaitValueEq, 0x7FFFFFFF, 0x7FFF0000,
                                                    0x7FFFFFFF))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("NoMask_Gte",
                                        TEST_WAIT32(hipStreamWaitValueGte, 0x7FFF0001, 0x7FFF0000,
                                                    0x7FFF0010))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("NoMask_And",
                                        TEST_WAIT32(hipStreamWaitValueAnd, 0x70F0F0F0, 0x0F0F0F0F,
                                                    0X1F0F0F0F))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("NoMask_Nor",
                                        TEST_WAIT32(hipStreamWaitValueNor, 0x7AAAAAAA,
                                                    static_cast<int32_t>(0x85555555),
                                                    static_cast<int32_t>(0x9AAAAAAA)))

#undef DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32

#define DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64(suffix, test_t)                                    \
  TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_" + std::string(suffix)) {                        \
    testWait<int64_t, true>(test_t);                                                               \
  }                                                                                                \
  TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_" + std::string(suffix)) {                     \
    testWait<int64_t, false>(test_t);                                                              \
  }


// Using Mask
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("Mask_Gte_1",
                                        TEST_WAIT64(  // mask will ignore few MSB bits
                                            hipStreamWaitValueGte, 0x0000FFFFFFFFFFFF,
                                            0x000000007FFF0001, 0x7FFF00007FFF0000,
                                            0x000000007FFF0001))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("Mask_Gte_2",
                                        TEST_WAIT64(hipStreamWaitValueGte, 0xF, 0x4, 0x3, 0x6))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("Mask_Eq_1",
                                        TEST_WAIT64(  // mask will ignore few MSB bits
                                            hipStreamWaitValueEq, 0x0000FFFFFFFFFFFF,
                                            0x000000000FFF0001, 0x7FFF00000FFF0000,
                                            0x7F0000000FFF0001))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("Mask_Eq_2",
                                        TEST_WAIT64(hipStreamWaitValueEq, 0xFF, 0x11, 0x25, 0x11))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("Mask_And",
                                        TEST_WAIT64(  // mask will discard bits 8 to 11
                                            hipStreamWaitValueAnd, 0xFF, 0xF4A, 0xF35, 0X02))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("Mask_Nor_1",
                                        TEST_WAIT64(  // mask is set to ignore the sign bit.
                                            hipStreamWaitValueNor, 0x7FFFFFFFFFFFFFFF,
                                            0x7FFFFFFFFFFFF247, 0x7FFFFFFFFFFFFdbd,
                                            0x7FFFFFFFFFFFFdb5))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("Mask_Nor_2",
                                        TEST_WAIT64(  // mask is set to apply NOR for bits 0 to 3.
                                            hipStreamWaitValueNor, 0xF, 0x7E, 0x7D, 0x76))

DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("NoMask_Gte",
                                        TEST_WAIT64(hipStreamWaitValueGte, 0x7FFFFFFFFFFF0001,
                                                    0x7FFFFFFFFFFF0000, 0x7FFFFFFFFFFF0001))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("NoMask_Eq",
                                        TEST_WAIT64(hipStreamWaitValueEq, 0x7FFFFFFFFFFFFFFF,
                                                    0x7FFFFFFF0FFF0000, 0x7FFFFFFFFFFFFFFF))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("NoMask_And",
                                        TEST_WAIT64(hipStreamWaitValueAnd, 0x70F0F0F0F0F0F0F0,
                                                    0x0F0F0F0F0F0F0F0F, 0X1F0F0F0F0F0F0F0F))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64("NoMask_Nor",
                                        TEST_WAIT64(hipStreamWaitValueNor, 0x4724724747247247,
                                                    static_cast<int64_t>(0xbddbddbdbddbddbd),
                                                    static_cast<int64_t>(0xbddbddbdbddbddb3)))
#undef DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64

#endif

// Negative Tests
TEST_CASE("Unit_hipStreamValue_Negative_InvalidMemory") {

#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-96");
  return;
#endif

  hipStream_t stream{nullptr};

  HIP_CHECK(hipStreamCreate(&stream));

  REQUIRE(stream != nullptr);

  // Allocate Host Memory
  auto hostPtr32 = std::unique_ptr<uint32_t>(new uint32_t(1));
  auto hostPtr64 = std::unique_ptr<uint64_t>(new uint64_t(1));

  // Register Host Memory
  HIP_CHECK(hipHostRegister(hostPtr32.get(), sizeof(int32_t), 0));
  HIP_CHECK(hipHostRegister(hostPtr64.get(), sizeof(int64_t), 0));

  // Set dummy data
  *hostPtr64 = 0x0;
  *hostPtr32 = 0x0;

  auto compareOp = hipStreamWaitValueGte;

  // Memory pointer negative tests

  INFO("Testing Invalid Memory Pointer for hipStreamWriteValue32");
  NEG_TEST_ERROR_CHECK(Write, 32, hipErrorNotSupported, stream, nullptr, 0, writeFlag)

  INFO("Testing Invalid Memory Pointer for hipStreamWriteValue64");
  NEG_TEST_ERROR_CHECK(Write, 64, hipErrorNotSupported, stream, nullptr, 0, writeFlag)

  INFO("Testing Invalid Memory Pointer for hipStreamWaitValue32");
  NEG_TEST_ERROR_CHECK(Wait, 32, hipErrorNotSupported, stream, nullptr, 0, compareOp)

  INFO("Testing Invalid Memory Pointer for hipStreamWaitValue64");
  NEG_TEST_ERROR_CHECK(Wait, 64, hipErrorNotSupported, stream, nullptr, 0, compareOp)

  // Cleanup
  HIP_CHECK(hipHostUnregister(hostPtr32.get()));
  HIP_CHECK(hipHostUnregister(hostPtr64.get()));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipStreamWaitValue_Negative_InvalidFlag") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-96");
  return;
#endif

  hipStream_t stream{nullptr};

  HIP_CHECK(hipStreamCreate(&stream));

  REQUIRE(stream != nullptr);

  // Allocate Host Memory
  auto hostPtr32 = std::unique_ptr<uint32_t>(new uint32_t(1));
  auto hostPtr64 = std::unique_ptr<uint64_t>(new uint64_t(1));

  // Register Host Memory
  HIP_CHECK(hipHostRegister(hostPtr32.get(), sizeof(int32_t), 0));
  HIP_CHECK(hipHostRegister(hostPtr64.get(), sizeof(int64_t), 0));

  // Set dummy data
  *hostPtr64 = 0x0;
  *hostPtr32 = 0x0;

  /* EXSWCPHIPT-96 */
  INFO("Testing Invalid flag for hipStreamWaitValue32");
  NEG_TEST_ERROR_CHECK(Wait, 32, hipErrorNotSupported, stream, hostPtr32.get(), 0, -1)
  INFO("Testing Invalid flag for hipStreamWaitValue64");
  NEG_TEST_ERROR_CHECK(Wait, 64, hipErrorNotSupported, stream, hostPtr64.get(), 0, -1)

  // Cleanup
  HIP_CHECK(hipHostUnregister(hostPtr32.get()));
  HIP_CHECK(hipHostUnregister(hostPtr64.get()));
  HIP_CHECK(hipStreamDestroy(stream));
}

#undef NEG_TEST_ERROR_CHECK
