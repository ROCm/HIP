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

// value used for memset operations
constexpr int testValue = 0x11;

enum class allocType { deviceMalloc, hostMalloc, hostRegisted, devRegistered };
enum class memSetType {
  hipMemset,
  hipMemsetD8,
  hipMemsetD16,
  hipMemsetD32,
  hipMemset2D,
  hipMemset3D
};

// helper struct containing vars needed for 2D and 3D memset Testing
struct MultiDData {
  size_t width{};
  // set to 0 for 1D
  size_t height{};
  // set to 0 for 2D
  size_t depth{};
  size_t pitch{};
};

// set of helper functions to tidy the nested switch statements
template <typename T>
static std::pair<T*,T*> deviceMallocHelper(memSetType memType, size_t dataW, size_t dataH, size_t dataD,
                             size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr{};
  switch (memType) {
    case memSetType::hipMemset3D: {
      hipPitchedPtr pitchedAPtr{};
      hipExtent extent;
      extent.width = dataW * elementSize;
      extent.height = dataH;
      extent.depth = dataD;

      pitchedAPtr =
          make_hipPitchedPtr(aPtr, extent.width, extent.width / elementSize, extent.height);
      HIP_CHECK(hipMalloc3D(&pitchedAPtr, extent));
      aPtr = reinterpret_cast<T*>(pitchedAPtr.ptr);
      dataPitch = pitchedAPtr.pitch;
      break;
    }

    case memSetType::hipMemset2D:
      HIP_CHECK(
          hipMallocPitch(reinterpret_cast<void**>(&aPtr), &dataPitch, dataW * elementSize, dataH));

      dataPitch = dataW * elementSize;
      break;

    default:
      HIP_CHECK(hipMalloc(&aPtr, sizeInBytes));
      dataPitch = dataW * elementSize;
      break;
  }
  return std::make_pair(aPtr, nullptr);
}

template <typename T>
static std::pair<T*, T*> hostMallocHelper(size_t dataW, size_t dataH, size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr;

  HIP_CHECK(hipHostMalloc(&aPtr, sizeInBytes));
  dataPitch = dataW * elementSize;

  return std::make_pair(aPtr, nullptr);
}

template <typename T>
static std::pair<T*, T*> hostRegisteredHelper(size_t dataW, size_t dataH, size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr = new T[dataW * dataH * dataD];

  HIP_CHECK(hipHostRegister(aPtr, sizeInBytes, hipHostRegisterDefault));

  dataPitch = dataW * elementSize;
  return std::make_pair(aPtr, nullptr);
}

template <typename T>
static std::pair<T*, T*> devRegisteredHelper(size_t dataW, size_t dataH, size_t dataD,
                                             size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr = new T[dataW * dataH * dataD];
  T* retPtr;

  HIP_CHECK(hipHostRegister(aPtr, sizeInBytes, hipHostRegisterDefault));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&retPtr), aPtr, 0));

  dataPitch = dataW * elementSize;
  // keep the address of the host memory
  return std::make_pair(retPtr, aPtr);
}

// helper function to allocate memory and set it to a value.
// retunr a pair of pointers due to the device registered allocation case, we need to keep track of
// the pointer to host memory to be able to unregister and free it
template <typename T>
static std::pair<T*, T*> initMemory(allocType type, memSetType memType, MultiDData& data) {
  size_t dataH = data.height == 0 ? 1 : data.height;
  size_t dataD = data.depth == 0 ? 1 : data.depth;
  std::pair<T*, T*> retPtr{};
  // check different types of allocation
  switch (type) {
    case allocType::deviceMalloc:
      retPtr = deviceMallocHelper<T>(memType, data.width, dataH, dataD, data.pitch);
      break;

    case allocType::hostMalloc:
      retPtr = hostMallocHelper<T>(data.width, dataH, dataD, data.pitch);
      break;

    case allocType::hostRegisted:
      retPtr = hostRegisteredHelper<T>(data.width, dataH, dataD, data.pitch);
      break;

    case allocType::devRegistered:
      retPtr = devRegisteredHelper<T>(data.width, dataH, dataD, data.pitch);
      break;

    default:
      REQUIRE(false);
      break;
  }
  return retPtr;
}

// set of helper functions to tidy the nested switch statements
template <typename T>
static void deviceMallocCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW, size_t dataH,
                             size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  switch (memType) {
    case memSetType::hipMemset3D: {
      hipMemcpy3DParms params{};
      params.kind = hipMemcpyDeviceToHost;
      params.srcPos = make_hipPos(0, 0, 0);
      params.srcPtr = make_hipPitchedPtr(aPtr, dataPitch, dataW, dataH);
      params.dstPos = make_hipPos(0, 0, 0);
      params.dstPtr = make_hipPitchedPtr(hostMem, dataPitch, dataW, dataH);

      hipExtent extent;
      extent.width = dataPitch;
      extent.height = dataH;
      extent.depth = dataD;

      params.extent = extent;

      HIP_CHECK(hipMemcpy3D(&params));
      break;
    }

    case memSetType::hipMemset2D:
      HIP_CHECK(hipMemcpy2D(hostMem, dataW * elementSize, aPtr, dataPitch, dataW, dataH,
                            hipMemcpyDeviceToHost));
      break;

    default:
      HIP_CHECK(hipMemcpy(hostMem, aPtr, sizeInBytes, hipMemcpyDeviceToHost));
      break;
  }
}

template <typename T>
static void hostCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW, size_t dataH,
                     size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  hipMemcpy3DParms params{};
  switch (memType) {
    case memSetType::hipMemset3D: {
      params.kind = hipMemcpyHostToHost;
      params.srcPos = make_hipPos(0, 0, 0);
      params.dstPos = make_hipPos(0, 0, 0);
      params.srcPtr = make_hipPitchedPtr(aPtr, dataPitch, dataW, dataH);
      params.dstPtr = make_hipPitchedPtr(hostMem, dataW, dataW, dataH);

      hipExtent extent;
      extent.width = dataW;
      extent.height = dataH;
      extent.depth = dataD;

      params.extent = extent;

      HIP_CHECK(hipMemcpy3D(&params));
      break;
    }

    case memSetType::hipMemset2D:
      HIP_CHECK(hipMemcpy2D(hostMem, dataW * elementSize, aPtr, dataPitch, dataW, dataH,
                            hipMemcpyHostToHost));
      break;

    default:
      HIP_CHECK(hipMemcpy(hostMem, aPtr, sizeInBytes, hipMemcpyHostToHost));
      break;
  }
}

template <typename T>
static void devRegisteredCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW, size_t dataH,
                              size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);

  switch (memType) {
    case memSetType::hipMemset3D: {
      hipMemcpy3DParms params{};
      params.kind = hipMemcpyHostToHost;
      params.srcPos = make_hipPos(0, 0, 0);
      params.dstPos = make_hipPos(0, 0, 0);
      params.srcPtr = make_hipPitchedPtr(aPtr, dataPitch, dataW, dataH);
      params.dstPtr = make_hipPitchedPtr(hostMem, dataW, dataW, dataH);

      hipExtent extent;
      extent.width = dataW;
      extent.height = dataH;
      extent.depth = dataD;

      params.extent = extent;

      HIP_CHECK(hipMemcpy3D(&params));
      break;
    }

    case memSetType::hipMemset2D:
      HIP_CHECK(hipMemcpy2D(hostMem, dataW * elementSize, aPtr, dataPitch, dataW, dataH,
                            hipMemcpyDeviceToHost));
      break;

    default: {
      size_t sizeInBytes = elementSize * dataW * dataH * dataD;
      HIP_CHECK(hipMemcpy(hostMem, aPtr, sizeInBytes, hipMemcpyDeviceToHost));
      break;
    }
  }
}

// Copies device data to host and checks that each element is equal to the
// specified value
template <typename T>
void verifyData(T* aPtr, size_t value, MultiDData& data, allocType type, memSetType memType) {
  auto dataH = data.height == 0 ? 1 : data.height;
  auto dataD = data.depth == 0 ? 1 : data.depth;
  size_t sizeInBytes = data.pitch * dataH * dataD;
  std::unique_ptr<T[]> hostPtr = std::make_unique<T[]>(data.pitch * dataH * dataD / sizeof(T));
  switch (type) {
    case allocType::deviceMalloc:
      deviceMallocCopy(memType, aPtr, hostPtr.get(), data.width, dataH, dataD, data.pitch);
      break;
    case allocType::devRegistered:
      devRegisteredCopy(memType, aPtr, hostPtr.get(), data.width, dataH, dataD, data.pitch);
      break;
    default:  // host allocated or host registered memory
      hostCopy(memType, aPtr, hostPtr.get(), data.width, dataH, dataD, data.pitch);
      break;
  }


  size_t idx;
  bool allMatch = true;

  for (size_t k = 0; k < dataD; k++) {
    for (size_t j = 0; j < dataH; j++) {
      for (size_t i = 0; i < data.width; i++) {
        idx = data.pitch * dataH * k + data.pitch * j + i;
        CAPTURE(sizeInBytes, i, j, k, value, data.pitch, reinterpret_cast<long>(aPtr));
        allMatch = allMatch && static_cast<size_t>(hostPtr.get()[idx]) == value;
        if (!allMatch) REQUIRE(false);
      }
    }
  }
}

// macro to allow reuse of functions for testing versions of hipMemset
template <typename T>
void memsetCheck(T* aPtr, size_t value, memSetType memsetType, MultiDData& data, bool async = false,
                 hipStream_t stream = nullptr) {
  size_t dataW = data.width;
  size_t dataH = data.height == 0 ? 1 : data.height;
  size_t dataD = data.depth == 0 ? 1 : data.depth;
  size_t count = dataW * dataH * dataD;

  switch (memsetType) {
    case memSetType::hipMemset:
      if (async) {
        HIP_CHECK(hipMemsetAsync(aPtr, value, count, stream));
      } else {
        HIP_CHECK(hipMemset(aPtr, value, count));
      }
      break;

    case memSetType::hipMemsetD8:
      if (async) {
        HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(aPtr), value, count, stream));
      } else {
        HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(aPtr), value, count));
      }

      break;

    case memSetType::hipMemsetD16:
      if (async) {
        HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(aPtr), value, count, stream));
      } else {
        HIP_CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(aPtr), value, count));
      }
      break;

    case memSetType::hipMemsetD32:
      if (async) {
        HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(aPtr), value, count, stream));
      } else {
        HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(aPtr), value, count));
      }
      break;

    case memSetType::hipMemset2D:
      if (async) {
        HIP_CHECK(hipMemset2DAsync(aPtr, data.pitch, value, data.width, data.height, stream));
      } else {
        HIP_CHECK(hipMemset2D(aPtr, data.pitch, value, data.width, data.height));
      }
      break;

    case memSetType::hipMemset3D:
      hipExtent extent;
      extent.width = data.width;
      extent.height = data.height;
      extent.depth = data.depth;
      if (async) {
        HIP_CHECK(hipMemset3DAsync(make_hipPitchedPtr(aPtr, data.pitch, data.width, data.height),
                                   value, extent, stream));
      } else {
        HIP_CHECK(hipMemset3D(make_hipPitchedPtr(aPtr, data.pitch, data.width, data.height), value,
                              extent));
      }
      break;

    default:
      REQUIRE(false);
      break;
  }
}

template <typename T> void freeStuff(T* aPtr, allocType type) {
  switch (type) {
    case allocType::deviceMalloc:
      hipFree(aPtr);
      break;
    case allocType::hostMalloc:
      hipHostFree(aPtr);
      break;
    case allocType::hostRegisted:
      HIP_CHECK(hipHostUnregister(aPtr));
      delete[] aPtr;
      break;
    case allocType::devRegistered:
      HIP_CHECK(hipHostUnregister(aPtr));
      delete[] aPtr;
      break;
    default:
      REQUIRE(false);
      break;
  }
}

// Helper function to run tests for hipMemset allocation types
template <typename T>
void runTests(allocType type, memSetType memsetType, MultiDData data, hipStream_t stream) {
  bool async = GENERATE(true, false);
  CAPTURE(type, memsetType, data.width, data.height, data.depth, stream, async);
  std::pair<T*, T*> aPtr = initMemory<T>(type, memsetType, data);
  launchLongRunningKernel(300, stream);
  memsetCheck(aPtr.first, testValue, memsetType, data, async, stream);

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

template <typename T>
static void doMemsetTest(allocType mallocType, memSetType memset_type, MultiDData data) {
  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));

  runTests<T>(mallocType, memset_type, data, stream);

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
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
  doMemsetTest<char>(type, memset_type, data);
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

  doMemsetTest<TestType>(mallocType, memset_type, data);
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

  doMemsetTest<char>(mallocType, memset_type, data);
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

  doMemsetTest<char>(mallocType, memset_type, data);
}
