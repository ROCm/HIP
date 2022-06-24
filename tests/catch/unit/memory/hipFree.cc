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

// Amount of time kernel should wait
constexpr size_t delay = 50;


TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncDev", "", char, float, float2, float4) {
  TestType* devPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);
  HIP_CHECK(hipMalloc(&devPtr, sizeof(TestType) * size_mult));

  runKernelForMs(delay);
  // make sure device is busy
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(hipFree(devPtr));
  HIP_CHECK(hipStreamQuery(nullptr));
}

TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncHost", "", char, float, float2, float4) {
  TestType* hostPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);

  HIP_CHECK(hipHostMalloc(&hostPtr, sizeof(TestType) * size_mult));

  runKernelForMs(delay);
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
    runKernelForMs(delay);
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
    runKernelForMs(delay);
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
  runKernelForMs(delay);
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

// Freeing a invalid pointer with on host
TEST_CASE("Unit_hipFreeNegativeHost") {
  SECTION("NullPtr") { HIP_CHECK(hipHostFree(nullptr)); }
  SECTION("InvalidPtr") {
    char hostPtr;
    HIP_CHECK_ERROR(hipHostFree(&hostPtr), hipErrorInvalidValue);
  }
  SECTION("hipHostRegister") {
    char* hostPtr = new char;
    auto flag = GENERATE(hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped);
    HIP_CHECK(hipHostRegister((void*)hostPtr, sizeof(char), flag));
    HIP_CHECK_ERROR(hipHostFree(hostPtr), hipErrorInvalidValue);
    delete hostPtr;
  }
}

#if HT_NVIDIA
TEST_CASE("Unit_hipFreeNegativeArray") {
  DriverContext ctx;
  hipArray_t arrayPtr{};
  hiparray cuArrayPtr{};

  SECTION("ArrayFree") { HIP_CHECK(hipFreeArray(nullptr)); }
  SECTION("ArrayDestroy") {
    HIP_CHECK_ERROR(hipArrayDestroy(nullptr), hipErrorInvalidResourceHandle);
  }
}
#else

// Freeing a invalid pointer with array
TEST_CASE("Unit_hipFreeNegativeArray") {
  SECTION("ArrayFree") { HIP_CHECK_ERROR(hipFreeArray(nullptr), hipErrorInvalidValue); }
  SECTION("ArrayDestroy") { HIP_CHECK_ERROR(hipArrayDestroy(nullptr), hipErrorInvalidValue); }
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
    HIP_CHECK(hipArrayDestroy(arrayPtr));
    HIP_CHECK_ERROR(hipArrayDestroy(arrayPtr), hipErrorContextIsDestroyed);
  }
}

#endif


TEMPLATE_TEST_CASE("Unit_hipFreeMultiTDev", "", char, int, float2, float4) {
  std::vector<TestType*> ptrs(numAllocs);
  size_t allocSize = sizeof(TestType) * GENERATE(1, 32, 64, 128);

  for (auto& ptr : ptrs) {
    HIP_CHECK(hipMalloc(&ptr, allocSize));
  }

  std::vector<std::thread> threads;

  for (auto ptr : ptrs) {
    threads.emplace_back(([ptr] {
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
  std::vector<TestType*> ptrs(numAllocs);
  size_t allocSize = sizeof(TestType) * GENERATE(1, 32, 64, 128);

  for (auto& ptr : ptrs) {
    HIP_CHECK(hipHostMalloc(&ptr, allocSize));
  }

  std::vector<std::thread> threads;

  for (auto ptr : ptrs) {
    threads.emplace_back(([ptr] {
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
      threads.emplace_back(([ptr] {
        HIP_CHECK_THREAD(hipArrayDestroy(ptr));
        HIP_CHECK_THREAD(hipStreamQuery(nullptr));
      }));
    }
    for (auto& t : threads) {
      t.join();
    }
    HIP_CHECK_THREAD_FINALIZE();
  }

  SECTION("ArrayFree") {
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
        threads.emplace_back(([ptr] {
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