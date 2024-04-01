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
#include <memory>
#include <type_traits>

constexpr unsigned int writeFlag = 0;

template <typename UIntT, typename... Args> auto waitFunc(Args... args) {
  if constexpr (std::is_same<UIntT, uint32_t>::value) {
    return hipStreamWaitValue32(args...);
  } else {
    return hipStreamWaitValue64(args...);
  }
};

template <typename UIntT, typename... Args> auto writeFunc(Args... args) {
  if constexpr (std::is_same<UIntT, uint32_t>::value) {
    return hipStreamWriteValue32(args...);
  } else {
    return hipStreamWriteValue64(args...);
  }
};

// Random predefined 32 and 64 bit values
using value32_t = std::integral_constant<uint32_t, 0x70F0F0FF>;
using value64_t = std::integral_constant<uint64_t, 0x7FFF0000FFFF0000>;
template <typename UIntT>
using testValue =
    typename std::conditional<std::is_same<UIntT, uint32_t>::value, value32_t, value64_t>::type;

constexpr uint32_t DATA_INIT = 0x1234;
constexpr uint32_t DATA_UPDATE = 0X4321;

template <typename UIntT> struct TEST_WAIT {
  static_assert(std::is_same<UIntT, uint32_t>::value or std::is_same<UIntT, uint64_t>::value,
                "only implemented for 32 bit and 64 bit unsigned integers");
  unsigned int compareOp;
  UIntT mask = ~static_cast<UIntT>(0);
  UIntT waitValue;
  UIntT signalValueFail;
  UIntT signalValuePass;

  TEST_WAIT(unsigned int compareOp, UIntT waitValue, UIntT signalValueFail, UIntT signalValuePass)
      : compareOp{compareOp},
        waitValue{waitValue},
        signalValueFail{signalValueFail},
        signalValuePass{signalValuePass} {}

  TEST_WAIT(unsigned int compareOp, UIntT mask, UIntT waitValue, UIntT signalValueFail,
            UIntT signalValuePass)
      : compareOp{compareOp},
        mask{mask},
        waitValue{waitValue},
        signalValueFail{signalValueFail},
        signalValuePass{signalValuePass} {}
};

using TEST_WAIT32 = TEST_WAIT<uint32_t>;
using TEST_WAIT64 = TEST_WAIT<uint64_t>;

bool streamWaitValueSupported() {
  int device_num = 0;
  HIP_CHECK(hipGetDeviceCount(&device_num));
  for (int device_id = 0; device_id < device_num; ++device_id) {
    HIP_CHECK(hipSetDevice(device_id));
    int waitValueSupport = 0;
    auto getAttributeError = hipDeviceGetAttribute(
        &waitValueSupport, hipDeviceAttributeCanUseStreamWaitValue, device_id);
    if (getAttributeError != hipSuccess) {
      HipTest::HIP_SKIP_TEST("attribute not supported");
      return false;
    }
    if (waitValueSupport == 1) return true;
  }
  return false;
}

// The different types of memory that can be used with hipStream[Wait|Write]
enum class PtrType { HostPtr, DevicePtr, DevicePtrToHost, Signal };

// Helper class to expose the pointer that is used with hipStream[Write|Wait]Value and also store a
// unique pointer with the deleter to simplify cleanup
// Also includes functions to update and get the value directly
template <PtrType type, typename UIntT, typename UniquePtrWithDeleter> class TestPtr {
  // This stores the memory that must be deleted, as well as the deleter
  UniquePtrWithDeleter ptrToDelete;

 public:
  // The pointer that should be used with hipStream[Write|Wait]Value
  UIntT* ptr;

  TestPtr(UIntT* ptr, UniquePtrWithDeleter ptrToDelete)
      : ptrToDelete(std::move(ptrToDelete)), ptr(ptr) {}


  // directly retrieve the value from wherever it was allocated
  UIntT getValue(size_t offset = 0) {
    if constexpr (type == PtrType::Signal || type == PtrType::HostPtr ||
                  type == PtrType::DevicePtrToHost) {
      return ptrToDelete.get()[offset];
    } else {
      static_assert(type == PtrType::DevicePtr, "Expected DevicePtr");
      UIntT value;
      HIP_CHECK(hipMemcpy(&value, ptr + offset, sizeof(UIntT), hipMemcpyDeviceToHost));
      return value;
    }
  }

  // directly set the value wherever it was allocated
  void setValue(UIntT value, size_t offset = 0) {
    if constexpr (type == PtrType::Signal || type == PtrType::DevicePtrToHost ||
                  type == PtrType::HostPtr) {
      ptrToDelete.get()[offset] = value;
    } else {
      // hipMemcpy causes deadlock, so use hipStreamWriteValue
      static_assert(type == PtrType::DevicePtr, "Expected DevicePtr");
      hipStream_t stream;
      HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
      HIP_CHECK(writeFunc<UIntT>(stream, ptr + offset, value, writeFlag));
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipStreamDestroy(stream));
    }
  }
};

// required for the static assert
template <PtrType> inline constexpr bool AMD_ACTIVE = HT_AMD == 1;

template <PtrType type, typename UIntT> auto allocMem() {
  constexpr std::size_t arraySize = 1024;
  if constexpr (type == PtrType::Signal) {
    static_assert(std::is_same<UIntT, uint64_t>::value,
                  "signal memory should only be used with 64bit memory");

    // Allocate Signal Memory
    uint64_t* signalPtr{};

    static_assert(AMD_ACTIVE<type>,
                  "nvidia backend compiler doesn't like hipExtMallocWithFlags, even in this "
                  "constexpr branch");
#if HT_AMD
    // 8 is the only acceptable size
    HIP_CHECK(
        hipExtMallocWithFlags(reinterpret_cast<void**>(&signalPtr), 8, hipMallocSignalMemory));
#endif

    // Init Memory
    *signalPtr = 0;

    auto freeStuff = [](uint64_t* sPtr) { HIP_CHECK(hipFree(sPtr)); };
    return TestPtr<type, UIntT, std::unique_ptr<uint64_t, decltype(freeStuff)>>{
        signalPtr, std::unique_ptr<uint64_t, decltype(freeStuff)>(signalPtr, freeStuff)};
  } else if constexpr (type == PtrType::DevicePtrToHost) {
    auto hostPtr = new UIntT[arraySize];

    // Register Host Memory
    HIP_CHECK(hipHostRegister(hostPtr, sizeof(UIntT) * arraySize, 0));

    // Init memory
    std::fill(hostPtr, hostPtr + arraySize, 0);

    UIntT* devicePtr;
    // Test writing device pointer
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&devicePtr), hostPtr, 0));
    auto freeStuff = [](UIntT* ptr) {
      HIP_CHECK(hipHostUnregister(ptr));
      delete[] ptr;
    };

    return TestPtr<type, UIntT, std::unique_ptr<UIntT[], decltype(freeStuff)>>{
        devicePtr, std::unique_ptr<UIntT[], decltype(freeStuff)>(hostPtr, freeStuff)};
  } else if constexpr (type == PtrType::HostPtr) {
    auto hostPtr = new UIntT[arraySize];

    // Register Host Memory
    HIP_CHECK(hipHostRegister(hostPtr, sizeof(UIntT) * arraySize, 0));

    // Init memory
    std::fill(hostPtr, hostPtr + arraySize, 0);

    auto freeStuff = [](UIntT* ptr) {
      HIP_CHECK(hipHostUnregister(ptr));
      delete[] ptr;
    };

    return TestPtr<type, UIntT, std::unique_ptr<UIntT[], decltype(freeStuff)>>{
        hostPtr, std::unique_ptr<UIntT[], decltype(freeStuff)>(hostPtr, freeStuff)};
  } else {
    static_assert(type == PtrType::DevicePtr, "Expected DevicePtr");
    UIntT* devicePtr;
    HIP_CHECK(hipMalloc(&devicePtr, sizeof(UIntT) * arraySize));
    HIP_CHECK(hipMemset(devicePtr, 0, sizeof(UIntT) * arraySize));
    auto freeStuff = [](UIntT* ptr) { HIP_CHECK(hipFree(ptr)); };
    return TestPtr<type, UIntT, std::unique_ptr<UIntT, decltype(freeStuff)>>{
        devicePtr, std::unique_ptr<UIntT, decltype(freeStuff)>(devicePtr, freeStuff)};
  }
}

// allows the creation of a list of offsets while avoiding it for signal memory
template <PtrType type> constexpr auto get_offsets() {
  if constexpr (type == PtrType::Signal) {
    return std::array<size_t, 1>{0};
  } else {
    return std::array<size_t, 6>{0, 1, 2, 3, 31, 1023};
  }
}

template <typename UIntT, PtrType ptrTypeValue> struct TestParams {
  using UIntType = UIntT;
  constexpr static PtrType ptrType = ptrTypeValue;
};

#if HT_AMD
TEMPLATE_TEST_CASE("Unit_hipStreamValue_Write", "", (TestParams<uint32_t, PtrType::HostPtr>),
                   (TestParams<uint32_t, PtrType::DevicePtr>),
                   (TestParams<uint32_t, PtrType::DevicePtrToHost>),
                   (TestParams<uint64_t, PtrType::HostPtr>),
                   (TestParams<uint64_t, PtrType::DevicePtr>),
                   (TestParams<uint64_t, PtrType::DevicePtrToHost>),
                   (TestParams<uint64_t, PtrType::Signal>)) {
#else
TEMPLATE_TEST_CASE("Unit_hipStreamValue_Write", "", (TestParams<uint32_t, PtrType::HostPtr>),
                   (TestParams<uint32_t, PtrType::DevicePtr>),
                   (TestParams<uint32_t, PtrType::DevicePtrToHost>),
                   (TestParams<uint64_t, PtrType::HostPtr>),
                   (TestParams<uint64_t, PtrType::DevicePtr>),
                   (TestParams<uint64_t, PtrType::DevicePtrToHost>)) {
#endif
  if (!streamWaitValueSupported()) {
    HipTest::HIP_SKIP_TEST("hipStreamWaitValue not supported on this device.");
    return;
  }

  using UIntT = typename TestType::UIntType;
  constexpr auto ptrType = TestType::ptrType;
  constexpr auto writeValue = testValue<UIntT>::value;

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));

  const auto offsets = get_offsets<ptrType>();
  const auto offset = GENERATE_COPY(from_range(std::begin(offsets), std::end(offsets)));

  CAPTURE(offset);

  // Allocate Host Memory
  auto ptr = allocMem<ptrType, UIntT>();
  UIntT* target = ptr.ptr + offset;
  HIP_CHECK(writeFunc<UIntT>(stream, target, writeValue, writeFlag));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(ptr.getValue(offset) == writeValue);

  // Cleanup
  HIP_CHECK(hipStreamDestroy(stream));
}

template <bool isBlocking, typename UIntT, typename TestPtr>
void syncAndCheckData(hipStream_t stream, UIntT* dataPtr, TestPtr signalPtr, size_t offset,
                      TEST_WAIT<UIntT> tc, std::array<hipEvent_t, 2>& events) {
  // Ensure first part of host memory is updated
  HIP_CHECK(hipEventSynchronize(events[0]));
  REQUIRE(dataPtr[0] == DATA_UPDATE);

  if constexpr (isBlocking) {
    // Ensure second part of host memory isn't updated yet
    HIP_CHECK_ERROR(hipEventQuery(events[1]), hipErrorNotReady);
    REQUIRE(dataPtr[1] == DATA_INIT);
    // Update value to release stream
    signalPtr.setValue(tc.signalValuePass, offset);
  }

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipEventQuery(events[1]));
  // Finally ensure that second part of host memory is updated
  REQUIRE(dataPtr[1] == DATA_UPDATE);
}


template <typename TestType, bool isBlocking>
void testWait(TEST_WAIT<typename TestType::UIntType> tc) {
  if (!streamWaitValueSupported()) {
    HipTest::HIP_SKIP_TEST("hipStreamWaitValue not supported on this device.");
    return;
  }
  using UIntT = typename TestType::UIntType;
  constexpr auto ptrType = TestType::ptrType;
  constexpr UIntT defaultMask = ~static_cast<UIntT>(0);

  // Initialize stream
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate Host Memory
  auto dataPtr = std::make_unique<UIntT[]>(2);
  // Register Host Memory
  HIP_CHECK(hipHostRegister(dataPtr.get(), sizeof(UIntT), 0));
  HIP_CHECK(hipHostRegister(dataPtr.get() + 1, sizeof(UIntT), 0));
  std::fill(dataPtr.get(), dataPtr.get() + 2, DATA_INIT);

  std::array<hipEvent_t, 2> events;
  HIP_CHECK(hipEventCreate(&events[0]));
  HIP_CHECK(hipEventCreate(&events[1]));


  const auto offsets = get_offsets<ptrType>();
  const auto offset = GENERATE_COPY(from_range(std::begin(offsets), std::end(offsets)));

  auto waitPtr = allocMem<ptrType, UIntT>();
  UIntT* const target = waitPtr.ptr + offset;
  waitPtr.setValue(isBlocking ? tc.signalValueFail : tc.signalValuePass, offset);

  HIP_CHECK(writeFunc<UIntT>(stream, &(dataPtr.get()[0]), DATA_UPDATE, writeFlag));
  HIP_CHECK(hipEventRecord(events[0], stream));

  if (tc.mask != defaultMask) {
    HIP_CHECK(waitFunc<UIntT>(stream, target, tc.waitValue, tc.compareOp, tc.mask));
  } else {
    HIP_CHECK(waitFunc<UIntT>(stream, target, tc.waitValue, tc.compareOp));
  }

  HIP_CHECK(writeFunc<UIntT>(stream, &(dataPtr.get()[1]), DATA_UPDATE, writeFlag));

  HIP_CHECK(hipEventRecord(events[1], stream));

  syncAndCheckData<isBlocking>(stream, dataPtr.get(), std::move(waitPtr), offset, tc, events);

  // Cleanup
  HIP_CHECK(hipEventDestroy(events[0]));
  HIP_CHECK(hipEventDestroy(events[1]));
  HIP_CHECK(hipHostUnregister(dataPtr.get()));
  HIP_CHECK(hipHostUnregister(dataPtr.get() + 1));
  HIP_CHECK(hipStreamDestroy(stream));
}

// TEMPLATE_TEST_CASE wasn't working within a macro, so sections were used instead
#define DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32(suffix, test_t)                                    \
  TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_" + std::string(suffix)) {                        \
    SECTION("HostPtr") { testWait<TestParams<uint32_t, PtrType::HostPtr>, true>(test_t); }         \
    SECTION("DevicePtr") { testWait<TestParams<uint32_t, PtrType::DevicePtr>, true>(test_t); }     \
    SECTION("DevicePtrToHost") {                                                                   \
      testWait<TestParams<uint32_t, PtrType::DevicePtrToHost>, true>(test_t);                      \
    }                                                                                              \
  }                                                                                                \
  TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_" + std::string(suffix)) {                     \
    SECTION("HostPtr") { testWait<TestParams<uint32_t, PtrType::HostPtr>, false>(test_t); }        \
    SECTION("DevicePtr") { testWait<TestParams<uint32_t, PtrType::DevicePtr>, false>(test_t); }    \
    SECTION("DevicePtrToHost") {                                                                   \
      testWait<TestParams<uint32_t, PtrType::DevicePtrToHost>, false>(test_t);                     \
    }                                                                                              \
  }


// Using Mask
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Gte",
                                        TEST_WAIT32(hipStreamWaitValueGte, 0xF, 0x4, 0x3, 0x6))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Eq_1",
                                        TEST_WAIT32(  // mask will ignore few MSB bits
                                            hipStreamWaitValueEq, 0x0000FFFF, 0x00000001,
                                            0x0FFF0000, 0x0FFF0001))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_Eq_2",
                                        TEST_WAIT32(hipStreamWaitValueEq, 0xFF, 0x11, 0x25, 0x11))
DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32("Mask_And",
                                        TEST_WAIT32(  // mask will discard bits 8 to 11
                                            hipStreamWaitValueAnd, 0xFF, 0xF4A, 0xF35, 0X02))

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
                                        TEST_WAIT32(hipStreamWaitValueNor, 0x7AAAAAAA, 0x85555555,
                                                    0x9AAAAAAA))

#undef DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT32

#if HT_AMD
// TEMPLATE_TEST_CASE wasn't working within a macro, so sections were used instead
#define DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64(suffix, test_t)                                    \
  TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_" + std::string(suffix)) {                        \
    SECTION("HostPtr") { testWait<TestParams<uint64_t, PtrType::HostPtr>, true>(test_t); }         \
    SECTION("DevicePtr") { testWait<TestParams<uint64_t, PtrType::DevicePtr>, true>(test_t); }     \
    SECTION("DevicePtrToHost") {                                                                   \
      testWait<TestParams<uint64_t, PtrType::DevicePtrToHost>, true>(test_t);                      \
    }                                                                                              \
    SECTION("Signal") { testWait<TestParams<uint64_t, PtrType::Signal>, true>(test_t); }           \
  }                                                                                                \
  TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_" + std::string(suffix)) {                     \
    SECTION("HostPtr") { testWait<TestParams<uint64_t, PtrType::HostPtr>, false>(test_t); }        \
    SECTION("DevicePtr") { testWait<TestParams<uint64_t, PtrType::DevicePtr>, false>(test_t); }    \
    SECTION("DevicePtrToHost") {                                                                   \
      testWait<TestParams<uint64_t, PtrType::DevicePtrToHost>, false>(test_t);                     \
    }                                                                                              \
    SECTION("Signal") { testWait<TestParams<uint64_t, PtrType::Signal>, false>(test_t); }          \
  }
#else
#define DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64(suffix, test_t)                                    \
  TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_" + std::string(suffix)) {                        \
    SECTION("HostPtr") { testWait<TestParams<uint64_t, PtrType::HostPtr>, true>(test_t); }         \
    SECTION("DevicePtr") { testWait<TestParams<uint64_t, PtrType::DevicePtr>, true>(test_t); }     \
    SECTION("DevicePtrToHost") {                                                                   \
      testWait<TestParams<uint64_t, PtrType::DevicePtrToHost>, true>(test_t);                      \
    }                                                                                              \
  }                                                                                                \
  TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_" + std::string(suffix)) {                     \
    SECTION("HostPtr") { testWait<TestParams<uint64_t, PtrType::HostPtr>, false>(test_t); }        \
    SECTION("DevicePtr") { testWait<TestParams<uint64_t, PtrType::DevicePtr>, false>(test_t); }    \
    SECTION("DevicePtrToHost") {                                                                   \
      testWait<TestParams<uint64_t, PtrType::DevicePtrToHost>, false>(test_t);                     \
    }                                                                                              \
  }
#endif


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
                                                    0xbddbddbdbddbddbd, 0xbddbddbdbddbddb3))
#undef DEFINE_STREAM_WAIT_VAL_TEST_CASES_INT64

// Negative Tests
TEST_CASE("Unit_hipStreamValue_Negative_InvalidMemory") {
  if (!streamWaitValueSupported()) {
    HipTest::HIP_SKIP_TEST("hipStreamWaitValue not supported on this device.");
    return;
  }

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);

  const auto compareOp = hipStreamWaitValueGte;
  const auto expectedError = hipErrorInvalidValue;

  // Memory pointer negative tests
  SECTION("Invalid Memory Pointer for hipStreamWriteValue32") {
    HIP_CHECK_ERROR(hipStreamWriteValue32(stream, nullptr, 0, writeFlag), expectedError);
  }
  SECTION("Invalid Memory Pointer for hipStreamWriteValue64") {
    HIP_CHECK_ERROR(hipStreamWriteValue64(stream, nullptr, 0, writeFlag), expectedError);
  }
  SECTION("Invalid Memory Pointer for hipStreamWaitValue32") {
    HIP_CHECK_ERROR(hipStreamWaitValue32(stream, nullptr, 0, compareOp), expectedError);
  }
  SECTION("Invalid Memory Pointer for hipStreamWaitValue32") {
    HIP_CHECK_ERROR(hipStreamWaitValue64(stream, nullptr, 0, compareOp), expectedError);
  }

  // Cleanup
  HIP_CHECK(hipStreamDestroy(stream));
}

TEMPLATE_TEST_CASE("Unit_hipStreamValue_Negative_UninitializedStream", "", uint32_t, uint64_t) {
  if (!streamWaitValueSupported()) {
    HipTest::HIP_SKIP_TEST("hipStreamWaitValue not supported on this device.");
    return;
  }

  hipStream_t stream{reinterpret_cast<hipStream_t>(0xFFFF)};

  // Allocate Host Memory
  auto hostPtr = std::make_unique<TestType>();

  // Register Host Memory
  HIP_CHECK(hipHostRegister(hostPtr.get(), sizeof(TestType), 0));

  // Set dummy data
  *hostPtr = 0x0;

  const auto compareOp = hipStreamWaitValueGte;
  const auto expectedError = hipErrorContextIsDestroyed;

  // Stream handle negative tests
  SECTION("Invalid Stream handle for hipStreamWriteValue") {
    HIP_CHECK_ERROR(writeFunc<TestType>(stream, hostPtr.get(), 0, writeFlag), expectedError);
  }

  SECTION("Invalid Stream handle for hipStreamWaitValue") {
    HIP_CHECK_ERROR(waitFunc<TestType>(stream, hostPtr.get(), 0, compareOp), expectedError);
  }

  // Cleanup
  HIP_CHECK(hipHostUnregister(hostPtr.get()));
}

TEMPLATE_TEST_CASE("Unit_hipStreamValue_Negative_InvalidFlag", "", uint32_t, uint64_t) {
  if (!streamWaitValueSupported()) {
    HipTest::HIP_SKIP_TEST("hipStreamWaitValue not supported on this device.");
    return;
  }

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);

  // Allocate Host Memory
  auto hostPtr = std::make_unique<TestType>();

  // Register Host Memory
  HIP_CHECK(hipHostRegister(hostPtr.get(), sizeof(TestType), 0));

  // Set dummy data
  *hostPtr = 0x0;

  HIP_CHECK_ERROR(waitFunc<TestType>(stream, hostPtr.get(), 0, -1), hipErrorInvalidValue);

  // Cleanup
  HIP_CHECK(hipHostUnregister(hostPtr.get()));
  HIP_CHECK(hipStreamDestroy(stream));
}
