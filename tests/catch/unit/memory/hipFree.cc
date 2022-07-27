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
#include "hipArrayCommon.hh"
#include "DriverContext.hh"

/*
 * This testcase verifies [ hipFree || hipFreeArray || hipFreeType::ArrayDestroy ||
 * hipFreeType::HostFree with hipHostMalloc ]
 * 1. Check that hipFree implicitly synchronises the device.
 * 2. Perform multiple allocations and then call hipFree on each pointer concurrently (from unique
 * threads) for different memory types and different allocation sizes.
 * 3. Pass nullptr as argument and check that no operation is performed and hipSuccess is returned.
 * 4. Pass an invalid ptr and check that hipErrorInvalidValue is returned.
 * 5. Call hipFree twice on the same pointer and check that the implementation handles the second
 * call correctly.
 * 6. HipFreeType::HostFree only:
 *    Try to free memory that has been registered with hipHostRegister and check that
 * hipErrorInvalidValue is returned.
 */


enum class FreeType { DevFree, ArrayFree, ArrayDestroy, HostFree };

__global__ void waitKernel(clock_t offset) {
  auto time = clock();
  while (clock() - time < offset) {
  }
}

static __global__ void clock_kernel(clock_t clock_count, size_t* co) {
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock() - start_clock;
  }
  *co = clock_offset;
}

// number of clocks the device is running at (device frequency)
static size_t ticksPerMillisecond = 0;

// helper function used to set the device frequency variable

static size_t findTicks() {
  hipDeviceProp_t prop;
  int device;
  size_t* clockOffset;
  HIP_CHECK(hipMalloc(&clockOffset, sizeof(size_t)));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  constexpr float milliseconds = 1000;
  constexpr float tolerance = 0.02 * milliseconds;

  clock_t devFreq = static_cast<clock_t>(prop.clockRate);  // in kHz
  clock_t time = devFreq * milliseconds;
  // Warmup
  hipLaunchKernelGGL(clock_kernel, dim3(1), dim3(1), 0, 0, time, clockOffset);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // try 10 times to find device frequency
  // after 10 attempts the result is likely good enough so just accept it
  size_t co = 0;

  for (int attempts = 10; attempts > 0; attempts--) {
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(clock_kernel, dim3(1), dim3(1), 0, 0, time, clockOffset);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventSynchronize(stop));

    float executionTime = 0;
    HIP_CHECK(hipEventElapsedTime(&executionTime, start, stop));

    HIP_CHECK(hipMemcpy(&co, clockOffset, sizeof(size_t), hipMemcpyDeviceToHost));
    if (executionTime >= (milliseconds - tolerance) &&
        executionTime <= (milliseconds + tolerance)) {
      // Timing is within accepted tolerance, break here
      break;
    } else {
      auto off = fabs(milliseconds - executionTime) / milliseconds;
      if (executionTime >= milliseconds) {
        time -= (time * off);
        --attempts;
      } else {
        time += (time * off);
        --attempts;
      }
    }
  }
  HIP_CHECK(hipFree(clockOffset));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return co;
}

// Launches a kernel which runs for specified amount of time
// Note: The current implementation uses HIP_CHECK which is not thread safe!
// Note the function assumes execution on a single device, if changing devices between calls of
// runKernelForMs set ticksPerMillisecond back to 0 (should be avoided as it slows down testing
// significantly)
static void runKernelForMs(size_t millis, hipStream_t stream = nullptr) {
  if (ticksPerMillisecond == 0) {
    ticksPerMillisecond = findTicks();
  }
  hipLaunchKernelGGL(waitKernel, dim3(1), dim3(1), 0, stream, ticksPerMillisecond * millis / 1000);
  HIP_CHECK(hipGetLastError());
}


// DevFree, ArrayFree, ArrayDestroy, HostFree
TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncDev", "", char, float, float2, float4) {
  TestType* devPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);
  HIP_CHECK(hipMalloc(&devPtr, sizeof(TestType) * size_mult));

  runKernelForMs(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(hipFree(devPtr));
  HIP_CHECK(hipStreamQuery(nullptr));
}

TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncHost", "", char, float, float2, float4) {
  TestType* hostPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);

  HIP_CHECK(hipHostMalloc(&hostPtr, sizeof(TestType) * size_mult));

  runKernelForMs(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(hipHostFree(hostPtr));
  HIP_CHECK(hipStreamQuery(nullptr));
}

#if HT_NVIDIA
TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncArray", "", char, float, float2, float4) {
  using vec_info = vector_info<TestType>;
  DriverContext ctx;


  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(32, 512, 1024);

  SECTION("ArrayFree") {
    hipArray_t arrayPtr{};
    hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArrayDefault));
    runKernelForMs(50);
    // make sure device is busy
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
    HIP_CHECK(hipFreeArray(arrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
  SECTION("ArrayDestroy") {
    hiparray cuArrayPtr{};

    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = width;
    cuDesc.Height = height;
    cuDesc.Format = vec_info::format;
    cuDesc.NumChannels = vec_info::size;
    HIP_CHECK(hipArrayCreate(&cuArrayPtr, &cuDesc));
    runKernelForMs(50);
    // make sure device is busy
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
    HIP_CHECK(hipArrayDestroy(cuArrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
}
#else  // AMD

TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncArray", "", char, float, float2, float4) {
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512, 1024);
  extent.height = GENERATE(0, 32, 128, 256, 512, 1024);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));
  runKernelForMs(50);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  // Second free segfaults
  SECTION("ArrayDestroy") {
    HIP_CHECK(hipArrayDestroy(arrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
  SECTION("ArrayFree") {
    HIP_CHECK(hipFreeArray(arrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
}

#endif

// Freeing a invalid pointer with on device
TEST_CASE("Unit_hipFreeNegativeDev") {
  SECTION("InvalidPtr") {
    char value;
    HIP_CHECK_ERROR(hipFree(&value), hipErrorInvalidValue);
  }
  SECTION("NullPtr") { HIP_CHECK(hipFree(nullptr)); }
}

// Freeing a invalid pointer with on device
TEST_CASE("Unit_hipFreeNegativeHost") {
  char* hostPtr{nullptr};
  SECTION("NullPtr") { HIP_CHECK(hipHostFree(nullptr)); }
  SECTION("InvalidPtr") {
    hostPtr = new (char);
    HIP_CHECK_ERROR(hipHostFree(hostPtr), hipErrorInvalidValue);
  }
  SECTION("hipHostRegister") {
    hostPtr = new char;
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped);
    HIP_CHECK(hipHostRegister((void*)hostPtr, sizeof(char), flag));
    HIP_CHECK_ERROR(hipHostFree(hostPtr), hipErrorInvalidValue);
  }
  delete hostPtr;
}

#if HT_NVIDIA
TEST_CASE("Unit_hipFreeNegativeArray") {
  DriverContext ctx;
  hipArray_t arrayPtr{};
  hiparray cuArrayPtr{};

  SECTION("InvalidPtr") {
    arrayPtr = static_cast<hipArray*>(malloc(sizeof(char)));
    cuArrayPtr = static_cast<hiparray>(malloc(sizeof(char)));

    SECTION("ArrayFree") { HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed); }
    SECTION("ArrayDestroy") {
      HIP_CHECK_ERROR(hipArrayDestroy(cuArrayPtr), hipErrorContextIsDestroyed);
    }
    free(arrayPtr);
    free(cuArrayPtr);
  }
  SECTION("NullPtr") {
    arrayPtr = nullptr;
    cuArrayPtr = nullptr;
    SECTION("ArrayFree") { HIP_CHECK(hipFreeArray(arrayPtr)); }
    SECTION("ArrayDestroy") {
      HIP_CHECK_ERROR(hipArrayDestroy(cuArrayPtr), hipErrorInvalidResourceHandle);
    }
  }
}
#else

// Freeing a invalid pointer with array
TEST_CASE("Unit_hipFreeNegativeArray") {
  hipArray_t arrayPtr{};

  SECTION("InvalidPtr") {
    arrayPtr = static_cast<hipArray*>(malloc(sizeof(char)));

    SECTION("ArrayFree") { HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed); }
    SECTION("ArrayDestroy") {
      HIP_CHECK_ERROR(hipArrayDestroy(arrayPtr), hipErrorContextIsDestroyed);
    }
  }

  SECTION("NullPtr") {
    arrayPtr = nullptr;
    SECTION("ArrayFree") { HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorInvalidValue); }
    SECTION("ArrayDestroy") { HIP_CHECK_ERROR(hipArrayDestroy(arrayPtr), hipErrorInvalidValue); }
  }

  free(arrayPtr);
}

#endif


TEST_CASE("Unit_hipFreeDoubleDevice") {
  size_t width = GENERATE(32, 512, 1024);
  char* ptr{};
  size_t size_mult = width;
  HIP_CHECK(hipMalloc(&ptr, sizeof(char) * size_mult));

  HIP_CHECK(hipFree(ptr));
  HIP_CHECK_ERROR(hipFree(ptr), hipErrorInvalidValue);
}
TEST_CASE("Unit_hipFreeDoubleHost") {
  size_t width = GENERATE(32, 512, 1024);
  char* ptr{};
  size_t size_mult = width;

  HIP_CHECK(hipHostMalloc(&ptr, sizeof(char) * size_mult));

  HIP_CHECK(hipHostFree(ptr));
  HIP_CHECK_ERROR(hipHostFree(ptr), hipErrorInvalidValue);
}

#if HT_NVIDIA
TEST_CASE("Unit_hipFreeDoubleArrayFree") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  HIP_CHECK(hipFreeArray(arrayPtr));
  HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed);
}

TEST_CASE("Unit_hipFreeDoubleArrayDestroy") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;
  using vec_info = vector_info<char>;

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  DriverContext ctx{};

  hiparray ArrayPtr{};
  HIP_ARRAY_DESCRIPTOR cuDesc;
  cuDesc.Width = width;
  cuDesc.Height = height;
  cuDesc.Format = vec_info::format;
  cuDesc.NumChannels = vec_info::size;
  HIP_CHECK(hipArrayCreate(&ArrayPtr, &cuDesc));
  HIP_CHECK(hipArrayDestroy(ArrayPtr));
  HIP_CHECK_ERROR(hipArrayDestroy(ArrayPtr), hipErrorContextIsDestroyed);
}

#else  // AMD

TEST_CASE("Unit_hipFreeDoubleArray") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  SECTION("ArrayFree") {
    HIP_CHECK(hipFreeArray(arrayPtr));
    HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed);
  }
  SECTION("ArrayDestroy") {
    HIP_CHECK(hipArrayDestroy(ArrayPtr));
    HIP_CHECK_ERROR(hipArrayDestroy(ArrayPtr), hipErrorContextIsDestroyed);
  }
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
    threads.emplace_back(std::thread([ptr] {
      HIP_CHECK_THREAD(hipFree(ptr));
      HIP_CHECK_THREAD(hipStreamQuery(nullptr));
    }));
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
    threads.emplace_back(std::thread([ptr] {
      HIP_CHECK_THREAD(hipHostFree(ptr));
      HIP_CHECK_THREAD(hipStreamQuery(nullptr));
    }));
  }

  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

#if HT_NVIDIA
TEMPLATE_TEST_CASE("Unit_hipFreeMultiTArray", "", char, int, float2, float4) {
  using vec_info = vector_info<TestType>;
  constexpr size_t numAllocs = 10;
  size_t width = GENERATE(32, 128, 256, 512, 1024);
  size_t height = GENERATE(32, 128, 256, 512, 1024);
  DriverContext ctx;
  std::vector<std::thread> threads;


  SECTION("ArrayDestroy") {
    std::vector<hiparray> ptrs(numAllocs);
    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = width;
    cuDesc.Height = height;
    cuDesc.Format = vec_info::format;
    cuDesc.NumChannels = vec_info::size;
    for (auto& ptr : ptrs) {
      HIP_CHECK(hipArrayCreate(&ptr, &cuDesc));
    }


    for (auto& ptr : ptrs) {
      threads.emplace_back(std::thread([ptr] {
        HIP_CHECK_THREAD(hipArrayDestroy(ptr));
        HIP_CHECK_THREAD(hipStreamQuery(nullptr));
      }));
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

    for (auto ptr : ptrs) {
      SECTION("ArrayFree") {
        threads.emplace_back(std::thread([ptr] {
          HIP_CHECK_THREAD(hipFreeArray(ptr));
          HIP_CHECK_THREAD(hipStreamQuery(nullptr));
        }));
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
  using vec_info = vector_info<TestType>;
  constexpr size_t numAllocs = 10;
  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512, 1024);
  extent.height = GENERATE(0, 32, 128, 256, 512, 1024);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  std::vector<std::thread> threads;

  SECTION("ArrayFree") {
    std::vector<hipArray_t> ptrs(numAllocs);
    for (auto& ptr : ptrs) {
      HIP_CHECK(hipMallocArray(&ptr, &desc, extent.width, extent.height, hipArrayDefault));
      threads.emplace_back([ptr] {
        HIP_CHECK_THREAD(hipFreeArray(ptr));
        HIP_CHECK_THREAD(hipStreamQuery(nullptr));
      });
    }
  }
  SECTION("ArrayDestroy") {
    std::vector<hiparray> cuArrayPtrs(numAllocs);

    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = extent.width;
    cuDesc.Height = extent.height;
    cuDesc.Format = vec_info::format;
    cuDesc.NumChannels = vec_info::size;
    for (auto ptr : cuArrayPtrs) {
      HIP_CHECK(hipArrayCreate(&ptr, &cuDesc));

      threads.emplace_back([ptr] {
        HIP_CHECK_THREAD(hipArrayDestroy(ptr));
        HIP_CHECK_THREAD(hipStreamQuery(nullptr));
      });
    }
  }
  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

#endif