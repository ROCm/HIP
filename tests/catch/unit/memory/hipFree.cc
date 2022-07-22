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
#include "DriverContext.hh"

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

__global__ void waitKernel(clock_t offset) {
  auto time = clock();
  while (clock() - time < offset) {
  }
}

static __global__ void clock_kernel(clock_t clock_count, size_t* clockOut, size_t* co = nullptr) {
  *clockOut = clock();
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock() - start_clock;
  }
  if (co != nullptr) *co = clock_offset;
}

class LongRunningKernel {
 public:
  static size_t Get() {
    static LongRunningKernel instance;
    return instance.clockTicksSec;
  }
  static void runKernelForMs(size_t ms, hipStream_t stream = nullptr) {
    auto ticks = LongRunningKernel::Get();
    hipLaunchKernelGGL(waitKernel, dim3(1), dim3(1), 0, stream, ticks * ms / 1000);
    HIP_CHECK(hipGetLastError());
  }

 private:
  LongRunningKernel() {
    hipDeviceProp_t prop;
    int device;
    size_t *clockFromKernel, *clockOffset;
    HIP_CHECK(hipMalloc(&clockFromKernel, sizeof(size_t)));
    HIP_CHECK(hipMalloc(&clockOffset, sizeof(size_t)));
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    constexpr float mseconds = 1000;
    constexpr float error = 0.02 * mseconds;

    clock_t devFreq = static_cast<clock_t>(prop.clockRate);  // in kHz
    clock_t time = devFreq * mseconds;


    while (1) {
      auto start = std::chrono::high_resolution_clock::now();
      hipLaunchKernelGGL(clock_kernel, dim3(1), dim3(1), 0, 0, time, clockFromKernel, clockOffset);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      auto stop = std::chrono::high_resolution_clock::now();
      auto result = std::chrono::duration<double, std::milli>(stop - start).count();
      size_t co = 0;
      HIP_CHECK(hipMemcpy(&co, clockOffset, sizeof(size_t), hipMemcpyDeviceToHost));
      if (result >= (mseconds - error) && result <= (mseconds + error)) {
        clockTicksSec = co;
        HIP_CHECK(hipFree(clockFromKernel));
        HIP_CHECK(hipFree(clockOffset));
        break;
      } else {
        auto off = fabs(mseconds - result) / mseconds;
        if (result >= mseconds) {
          time -= (time * off);
        } else {
          time += (time * off);
        }
      }
    }
  }

  size_t clockTicksSec;
};


// helper functions to release memory
template <typename T> hipError_t freeStuff(T ptr, FreeType type) {
  switch (type) {
    case DevFree:
      return hipFree(ptr);
      break;
    case HostFree:
      return hipHostFree(ptr);
      break;
    case ArrayFree: {
      auto arrPtr = reinterpret_cast<hipArray_t>(ptr);
      return hipFreeArray(arrPtr);
      break;
    }
    case ArrayDestroy: {
#if HT_NVIDIA
      auto arrPtr = reinterpret_cast<hiparray>(ptr);
      return hipArrayDestroy(arrPtr);
#else
      auto arrPtr = reinterpret_cast<hipArray_t>(ptr);
      return hipFreeArray(arrPtr);
#endif
      break;
    }
    default:
      return hipErrorIllegalState;
      break;
  }
}

// Helper function to check if work on device is done
template <typename T> hipError_t workIsDoneCheck(T ptr, FreeType fType) {
  auto error = hipSuccess;
  // free memory
  error = freeStuff(ptr, fType);
  if (error != hipSuccess) {
    printf("Breaking on free \n");
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

  LongRunningKernel::runKernelForMs(50);
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

  LongRunningKernel::runKernelForMs(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(workIsDoneCheck<TestType*>(hostPtr, HostFree));

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_NVIDIA
TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncArray", "", char, float, float2, float4) {
  DriverContext ctx;

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(32, 512, 1024);

  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};
  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));

  SECTION("ArrayFree") {
    hipArray_t arrayPtr{};
    hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArrayDefault));
    LongRunningKernel::runKernelForMs(50);
    // make sure device is busy
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
    // Second free segfaults
    HIP_CHECK(workIsDoneCheck<hipArray_t>(arrayPtr, ArrayFree));
  }
  SECTION("ArrayDestroy") {
    hiparray cuArrayPtr{};

    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = width;
    cuDesc.Height = height;
    cuDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
    cuDesc.NumChannels = 2;
    HIP_CHECK(hipArrayCreate(&cuArrayPtr, &cuDesc));
    LongRunningKernel::runKernelForMs(50);
    // make sure device is busy
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
    HIP_CHECK(workIsDoneCheck<hiparray>(cuArrayPtr, ArrayDestroy));
  }

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}
#else  // AMD

TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncArray", "", char, float, float2, float4) {
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
  LongRunningKernel::runKernelForMs(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  // Second free segfaults
  SECTION("ArrayDestroy") { HIP_CHECK(workIsDoneCheck<hipArray_t>(arrayPtr, ArrayDestroy)); }
  SECTION("ArrayFree") { HIP_CHECK(workIsDoneCheck<hipArray_t>(arrayPtr, ArrayFree)); }

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
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
  char* hostPtr{nullptr};
  SECTION("NullPtr") {
    hostPtr = nullptr;
    HIP_CHECK(freeStuff(hostPtr, HostFree));
  }
  SECTION("InvalidPtr") {
    hostPtr = new (char);
    HIP_CHECK_ERROR(freeStuff(hostPtr, HostFree), hipErrorInvalidValue);
  }
  SECTION("hipHostRegister") {
    hostPtr = new char;
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped);
    HIP_CHECK(hipHostRegister((void*)hostPtr, sizeof(char), flag));
    HIP_CHECK_ERROR(freeStuff(hostPtr, HostFree), hipErrorInvalidValue);
  }
}

#if HT_NVIDIA
TEST_CASE("Unit_hipFreeNegativeArray") {
  DriverContext ctx;
  hipArray_t arrayPtr{};
  hiparray cuArrayPtr{};

  SECTION("InvalidPtr") {
    arrayPtr = static_cast<hipArray*>(malloc(sizeof(char)));

    SECTION("ArrayFree") {
      HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayFree), hipErrorContextIsDestroyed);
    }
    SECTION("ArrayDestroy") {
      HIP_CHECK_ERROR(freeStuff(cuArrayPtr, ArrayDestroy), hipErrorInvalidResourceHandle);
    }
    free(arrayPtr);
  }
  SECTION("NullPtr") {
    arrayPtr = nullptr;
    SECTION("ArrayFree") { HIP_CHECK(freeStuff(arrayPtr, ArrayFree)); }
    SECTION("ArrayDestroy") {
      HIP_CHECK_ERROR(freeStuff(cuArrayPtr, ArrayDestroy), hipErrorInvalidResourceHandle);
    }
  }
}
#else

// Freeing a invalid pointer with array
TEST_CASE("Unit_hipFreeNegativeArray") {
  hipArray_t arrayPtr{};

  SECTION("InvalidPtr") {
    arrayPtr = static_cast<hipArray*>(malloc(sizeof(char)));

    SECTION("ArrayFree") {
      HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayFree), hipErrorContextIsDestroyed);
    }
    SECTION("ArrayDestroy") {
      HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayDestroy), hipErrorContextIsDestroyed);
    }
  }

  SECTION("NullPtr") {
    arrayPtr = nullptr;
    SECTION("ArrayFree") { HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayFree), hipErrorInvalidValue); }
    SECTION("ArrayDestroy") {
      HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayDestroy), hipErrorInvalidValue);
    }
  }
}

#endif


TEST_CASE("Unit_hipFreeDoubleDevice") {
  size_t width = GENERATE(32, 512, 1024);
  char* ptr{};
  size_t size_mult = width;
  HIP_CHECK(hipMalloc(&ptr, sizeof(char) * size_mult));

  HIP_CHECK(freeStuff(ptr, DevFree));
  HIP_CHECK_ERROR(freeStuff(ptr, DevFree), hipErrorInvalidValue);
}
TEST_CASE("Unit_hipFreeDoubleHost") {
  size_t width = GENERATE(32, 512, 1024);
  char* ptr{};
  size_t size_mult = width;

  HIP_CHECK(hipHostMalloc(&ptr, sizeof(char) * size_mult));

  HIP_CHECK(freeStuff(ptr, HostFree));
  HIP_CHECK_ERROR(freeStuff(ptr, HostFree), hipErrorInvalidValue);
}
TEST_CASE("Unit_hipFreeDoubleArrayFree") {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;
#endif
  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  HIP_CHECK(freeStuff(arrayPtr, ArrayFree));
  HIP_CHECK_ERROR(freeStuff(arrayPtr, ArrayFree), hipErrorContextIsDestroyed);
}
#if HT_NVIDIA
TEST_CASE("Unit_hipFreeDoubleArrayDestroy") {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;
#endif
  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  DriverContext ctx{};
  // HipTest::HIP_SKIP_TEST("EXSWCPHIPT-81");
  // return;

  hiparray ArrayPtr{};
  HIP_ARRAY_DESCRIPTOR cuDesc;
  cuDesc.Width = width;
  cuDesc.Height = height;
  cuDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  cuDesc.NumChannels = 2;
  HIP_CHECK(hipArrayCreate(&ArrayPtr, &cuDesc));
  HIP_CHECK(freeStuff(ArrayPtr, ArrayDestroy));
  HIP_CHECK_ERROR(freeStuff(ArrayPtr, ArrayDestroy), hipErrorContextIsDestroyed);
}
#endif


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

#if HT_NVIDIA
TEMPLATE_TEST_CASE("Unit_hipFreeMultiTArray", "", char, int, float2, float4) {
  constexpr size_t numAllocs = 10;
  size_t width = GENERATE(32, 128, 256, 512, 1024);
  size_t height = GENERATE(32, 128, 256, 512, 1024);
  DriverContext ctx;


  SECTION("ArrayDestroy") {
    std::vector<hiparray> ptrs(numAllocs);
    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = width;
    cuDesc.Height = height;
    cuDesc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
    cuDesc.NumChannels = 2;
    for (auto& ptr : ptrs) {
      HIP_CHECK(hipArrayCreate(&ptr, &cuDesc));
    }

    std::vector<std::thread> threads;

    for (auto& ptr : ptrs) {
      SECTION("ArrayDestroy") {
        threads.push_back(std::thread(
            [ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<hiparray>(ptr, ArrayDestroy)); }));
      }
    }
    for (auto& t : threads) {
      t.join();
    }
    HIP_CHECK_THREAD_FINALIZE();
  };

  SECTION("ArrayFree") {
    constexpr size_t numAllocs = 10;
    std::vector<hipArray_t> ptrs(numAllocs);
    hipExtent extent{};
    extent.width = width;
    extent.height = height;
    hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

    for (auto& ptr : ptrs) {
      HIP_CHECK(hipMallocArray(&ptr, &desc, extent.width, extent.height, hipArrayDefault));
    }

    std::vector<std::thread> threads;

    for (auto ptr : ptrs) {
      SECTION("ArrayFree") {
        threads.push_back(std::thread(
            [ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<hipArray_t>(ptr, ArrayFree)); }));
      }
      SECTION("ArrayDestroy") {
        threads.push_back(std::thread(
            [ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<hipArray_t>(ptr, ArrayDestroy)); }));
      }
    }
    for (auto& t : threads) {
      t.join();
    }
    HIP_CHECK_THREAD_FINALIZE();
  }
}
#else

TEMPLATE_TEST_CASE("Unit_hipFreeMultiTArray", "", char, int, float2, float4) {
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
    SECTION("ArrayDestroy") {
      threads.push_back(std::thread(
          [ptr]() { HIP_CHECK_THREAD(workIsDoneCheck<hipArray_t>(ptr, ArrayDestroy)); }));
    }
  }
  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

#endif
