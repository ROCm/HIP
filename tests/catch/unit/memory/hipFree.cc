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
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/*
 * This testcase verifies [ hipFree || hipFreeArray || hipArrayDestroy || hipHostFree with
 * hipHostMalloc ]
 * 1. Check that hipFree implicitly synchronises the device.
 * 2.Perform multiple allocations and then call hipFree on each pointer concurrently (from unique
 * threads) for different memory types and different allocation sizes.
 * 3. Pass nullptr as argument and check that no operation is performed and hipSuccess is returned.
 * 4. Pass an invalid ptr and check that hipErrorInvalidValue is returned.
 * 5. Call hipFree twice on the same pointer and check that the implementation handles the second
 * call correctly.
 * 6. HipHostFree only:
 *    Try to free memory that has been registered with hipHostRegister and check that
 * hipErrorInvalidValue is returned.
 */


enum FreeType { DevFree, ArrayFree, ArrayDestroy, HostFree };

// long running kernel (time based)
__global__ void longRunningKernel(clock_t clock_count) {
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock() - start_clock;
  }
}


// Helper function which gets the clock frequency of device
struct ClockRate {
 private:
  clock_t value;
  ClockRate() {
    hipDeviceProp_t prop;
    int device;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    value = static_cast<clock_t>(prop.clockRate);  // in kHz
  }

 public:
  static clock_t Get() {
    static ClockRate instance;
    return instance.value;
  }
};

// Helper function for long running kernel launch
void lauchLongRunningKernel(clock_t miliseconds) {
  auto devFreq = ClockRate::Get();
  auto time = devFreq * miliseconds;
  hipLaunchKernelGGL(longRunningKernel, dim3(1), dim3(1), 0, 0, reinterpret_cast<clock_t>(time));
}

// helper functions to release memory
template <typename T> hipError_t freeStuff(T ptr, FreeType type) {
  switch (type) {
    case DevFree:
      return hipFree(ptr);
      break;
    case HostFree:
      return hipHostFree(ptr);
      break;
    default:
      return hipErrorIllegalState;
      break;
  }
}

#if HT_NVIDIA
template <> hipError_t freeStuff<hipArray_t>(hipArray_t ptr, FreeType type) {
  switch (type) {
    case ArrayFree:
      return hipFreeArray(ptr);
      break;
    case ArrayDestroy:
      return hipErrorIllegalState;
      break;
    default:
      return hipErrorIllegalState;
      break;
  }
}

template <> hipError_t freeStuff<hiparray>(CUarray ptr, FreeType type) {
  switch (type) {
    case ArrayFree:
      return hipErrorIllegalState;
      break;
    case ArrayDestroy:
      return hipArrayDestroy(ptr);
      break;
    default:
      return hipErrorIllegalState;
      break;
  }
}
#else
template <> hipError_t freeStuff<hipArray_t>(hipArray_t ptr, FreeType type) {
  switch (type) {
    case ArrayFree:
      return hipFreeArray(ptr);
      break;
    case ArrayDestroy:
      return hipArrayDestroy(ptr);
      break;
    default:
      return hipErrorIllegalState;
      break;
  }
}
#endif

// Helper function to check if work on device is done
template <typename T> hipError_t workIsDoneCheck(T ptr, FreeType fType) {
  auto error = hipSuccess;
  // free memory
  error = freeStuff(ptr, fType);
  if (error != hipSuccess) {
    printf("Breaking on first free \n");
    return error;
  }
  // ensure memory is freed
  // SegFaults the when using hipFreeArray or hipArrayDestroy  </3
  error = freeStuff(ptr, fType);
  if (error != hipErrorInvalidValue) {
    printf("Breaking on second free \n");
    return error;
  }
  // verify synchronization
  error = hipStreamQuery(nullptr);
  if (error != hipSuccess) {
    printf("Breaking on hipStreamQuery free \n");
  }

  return error;
}

// DevFree, ArrayFree, ArrayDestroy, HostFree
TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncDev", "", char, float, float2, float4) {
  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};
  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));

  TestType* devPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);
  HIP_CHECK(hipMalloc(&devPtr, sizeof(TestType) * size_mult));

  lauchLongRunningKernel(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(workIsDoneCheck<TestType*>(devPtr, DevFree));
  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}

TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncHost", "", char, float, float2, float4) {
  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};
  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));
  TestType* hostPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);

  HIP_CHECK(hipHostMalloc(&hostPtr, sizeof(TestType) * size_mult));

  lauchLongRunningKernel(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(workIsDoneCheck<TestType*>(hostPtr, HostFree));

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}

TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncArray", "", char, float, float2, float4) {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-79");
  return;

  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};
  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));

  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512, 1024);
  extent.height = GENERATE(0, 32, 128, 256, 512, 1024);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));
  lauchLongRunningKernel(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  // Second free segfaults
#if HT_AMD
  SECTION("ArrayDestroy") { HIP_CHECK(workIsDoneCheck<hipArray_t>(arrayPtr, ArrayDestroy)); }
#endif
  SECTION("ArrayFree") { HIP_CHECK(workIsDoneCheck<hipArray_t>(arrayPtr, ArrayFree)); }

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}

// On Nvidia devices the CUarray is used when calling hipArrayDestroy
#if HT_NVIDIA
TEST_CASE("Unit_hipFreeImplicitSyncArrayD") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-81");
  return;
  hiparray cuArrayPtr{};
  CTX_CREATE()

  HIP_ARRAY_DESCRIPTOR cuDesc;
  cuDesc.Width = 32;
  cuDesc.Height = 32;
  cuDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  cuDesc.NumChannels = 2;
  HIP_CHECK(hipArrayCreate(&cuArrayPtr, &cuDesc));
  lauchLongRunningKernel(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  SECTION("ArrayDestroy") { HIP_CHECK(workIsDoneCheck<hiparray>(cuArrayPtr, ArrayDestroy)); }
  CTX_DESTROY()
}
#endif

// Freeing a invalid pointer with on device
TEST_CASE("Unit_hipFreeNegativeDev") {
  char* devPtr;
  SECTION("InvalidPtr") {
    devPtr = new (char);
    HIP_CHECK_ERROR(freeStuff(devPtr, DevFree), hipErrorInvalidValue);
    delete devPtr;
  }
  SECTION("NullPtr") {
    devPtr = nullptr;
    HIP_CHECK(freeStuff(devPtr, DevFree));
  }
}

// Freeing a invalid pointer with on device
TEST_CASE("Unit_hipFreeNegativeHost") {
#ifdef HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-82");
  return;
#endif
  char* hostPtr{nullptr};
  SECTION("hipHostRegister") {
    hostPtr = new char;
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped);
    HIP_CHECK(hipHostRegister((void*)hostPtr, sizeof(char), flag));
    HIP_CHECK_ERROR(freeStuff(hostPtr, HostFree), hipErrorInvalidValue);
    HIP_CHECK(hipHostUnregister(hostPtr));
    free(hostPtr);
  }
  SECTION("NullPtr") {
    hostPtr = nullptr;
    HIP_CHECK(freeStuff(hostPtr, HostFree));
  }
  SECTION("InvalidPtr") {
    hostPtr = new (char);
    HIP_CHECK_ERROR(freeStuff(hostPtr, HostFree), hipErrorInvalidValue);
    free(hostPtr);
  }
}

// Freeing a invalid pointer with array
TEST_CASE("Unit_hipFreeNegativeArray") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-80&81");
  return;

  CTX_CREATE()
  hipArray_t arrayPtr{};
#if HT_NVIDIA
  hiparray cuArrayPtr{};
#endif
  SECTION("InvalidPtr") {
    arrayPtr = static_cast<hipArray*>(malloc(sizeof(char)));

    SECTION("ArrayFree") {
      // on Nvidia returns  "context is destroyed" error 709
      // on AMD return success
      HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayFree), hipErrorInvalidValue);
    }
    SECTION("ArrayDestroy") {
#if HT_NVIDIA
      // on Nvidia returns "invalid resource handle" error 400
      HIP_CHECK_ERROR(freeStuff(cuArrayPtr, ArrayDestroy), hipErrorInvalidValue);
#else
      // on AMD return success
      HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayDestroy), hipErrorInvalidValue);
#endif
    }
    free(arrayPtr);
  }
  SECTION("NullPtr") {
    arrayPtr = nullptr;
    SECTION("ArrayFree") {
      // on AMD returns "invalid value" error 1
      HIP_CHECK(freeStuff(arrayPtr, ArrayFree));
    }
    SECTION("ArrayDestroy") {
#if HT_NVIDIA
      // on Nvidia returns "invalid resource handle" error 400
      HIP_CHECK(freeStuff(cuArrayPtr, ArrayDestroy));
#else
      // on AMD returns "invalid value" error 1
      HIP_CHECK(freeStuff(arrayPtr, ArrayDestroy));
#endif
    }
  }
  CTX_DESTROY()
}

// DevFree, ArrayFree, ArrayDestroy, HostFree Mutithreaded
TEMPLATE_TEST_CASE("Unit_hipFreeMultiTDev", "", char, int, float2, float4) {
  constexpr size_t numAllocs = 10;
  std::vector<TestType*> ptrs(numAllocs);
  size_t allocSize = sizeof(TestType) * GENERATE(1, 32, 64, 128);

  for (auto& ptr : ptrs) {
    HIP_CHECK(hipMalloc(&ptr, allocSize));
  }

  std::vector<std::thread> threads;

  for (auto ptr : ptrs) {
    threads.push_back(
        std::thread([ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<TestType*>(ptr, DevFree)); }));
  }

  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

TEMPLATE_TEST_CASE("Unit_hipFreeMultiTHost", "", char, int, float2, float4) {
  constexpr size_t numAllocs = 10;
  std::vector<TestType*> ptrs(numAllocs);
  size_t allocSize = sizeof(TestType) * GENERATE(1, 32, 64, 128);

  for (auto& ptr : ptrs) {
    HIP_CHECK(hipHostMalloc(&ptr, allocSize));
  }

  std::vector<std::thread> threads;

  for (auto ptr : ptrs) {
    threads.push_back(
        std::thread([ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<TestType*>(ptr, HostFree)); }));
  }

  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

TEMPLATE_TEST_CASE("Unit_hipFreeMultiTArray", "", char, int, float2, float4) {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-79&81");
  return;

  constexpr size_t numAllocs = 10;
  std::vector<hipArray_t> ptrs(numAllocs);
  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512, 1024);
  extent.height = GENERATE(0, 32, 128, 256, 512, 1024);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  for (auto& ptr : ptrs) {
    HIP_CHECK(hipMallocArray(&ptr, &desc, extent.width, extent.height, hipArrayDefault));
  }

  std::vector<std::thread> threads;

  for (auto ptr : ptrs) {
    SECTION("ArrayFree") {
      threads.push_back(
          std::thread([ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<hipArray_t>(ptr, ArrayFree)); }));
    }
#if HT_AMD
    SECTION("ArrayDestroy") {
      threads.push_back(std::thread(
          [ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<hipArray_t>(ptr, ArrayDestroy)); }));
    }
#endif
  }
  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

// On Nvidia devices the CUarray is used when calling hipArrayDestroy
#if HT_NVIDIA
TEST_CASE("Unit_hipFreeMultiTArrayDestroy") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-81");
  return;
  constexpr size_t numAllocs = 10;
  CTX_CREATE();
  std::vector<hiparray> ptrs(numAllocs);
  HIP_ARRAY_DESCRIPTOR cuDesc;
  cuDesc.Width = GENERATE(32, 128, 256, 512, 1024);
  cuDesc.Height = GENERATE(32, 128, 256, 512, 1024);
  cuDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  cuDesc.NumChannels = 2;
  for (auto& ptr : ptrs) {
    HIP_CHECK(hipArrayCreate(&ptr, &cuDesc));
  }

  std::vector<std::thread> threads;

  for (auto& ptr : ptrs) {
    SECTION("ArrayDestroy") {
      threads.push_back(
          std::thread([ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<hiparray>(ptr, ArrayDestroy)); }));
    }
  }
  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
  CTX_DESTROY();
}
#endif
