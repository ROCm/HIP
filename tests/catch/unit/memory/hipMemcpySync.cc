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
 * These testcases verify that synchronization behaviour for memcpy functions with respect to
 * the host.
 */

using namespace mem_utils;

// value used for memset operations
constexpr int testValue = 0x11;


/*
 * Set of helper functions handling the different cases for memcpy
 */

static inline hipMemcpyKind getMemcpyType(allocType type, bool fromHost) {
  hipMemcpyKind cpyType{};
  if (fromHost) {
    switch (type) {
      case allocType::deviceMalloc:
        cpyType = hipMemcpyHostToDevice;
        break;
      case allocType::devRegistered:
        cpyType = hipMemcpyHostToDevice;
        break;
      default:  // host
        cpyType = hipMemcpyHostToHost;
        break;
    }
  } else {
    switch (type) {
      case allocType::deviceMalloc:
        cpyType = hipMemcpyDeviceToDevice;
        break;
      case allocType::devRegistered:
        cpyType = hipMemcpyDeviceToDevice;
        break;
      default:  // host
        cpyType = hipMemcpyDeviceToHost;
        break;
    }
  }
  return cpyType;
}

template <typename T>
static inline void memcpyCheck(allocType type, memType memType, T* aPtr, MultiDData& data,
                               T* fillerData, bool async, hipStream_t stream,
                               bool fromHost) {
  auto cpyType = getMemcpyType(type, fromHost);
  auto sizeInBytes = data.pitch * data.getH() * data.getD() * sizeof(T);
  switch (memType) {
    case memType::hipMem:
      if (async) {
        HIP_CHECK(hipMemcpyAsync(aPtr + data.offset, fillerData, sizeInBytes, cpyType, stream));
      } else {
        HIP_CHECK(hipMemcpy(aPtr + data.offset, fillerData, sizeInBytes, cpyType));
      }
      break;
    case memType::hipMem2D:
      if (async) {
        HIP_CHECK(hipMemcpy2DAsync(aPtr + data.offset, data.pitch, fillerData, sizeInBytes,
                                   data.width, data.getH(), cpyType, stream));
      } else {
        HIP_CHECK(hipMemcpy2D(aPtr + data.offset, data.pitch, fillerData, sizeInBytes, data.width,
                              data.getH(), cpyType));
      }
      break;
    case memType::hipMem3D: {
      hipMemcpy3DParms params{};
      params.kind = cpyType;
      params.srcPos = make_hipPos(0, 0, 0);
      params.dstPos = make_hipPos(data.offset, data.offset, data.offset);
      params.srcPtr = make_hipPitchedPtr(fillerData, data.width, data.width, data.getH());
      params.dstPtr = make_hipPitchedPtr(aPtr, data.pitch, data.width, data.getH());
      hipExtent extent;
      extent.width = data.width * sizeof(T);
      extent.height = data.getH();
      extent.depth = data.getD();

      params.extent = extent;
      if (async) {
        HIP_CHECK(hipMemcpy3DAsync(&params, stream));
      } else {
        HIP_CHECK(hipMemcpy3D(&params));
      }
      break;
    }
    default:
      break;
  }
  if(fromHost) {
    delete[] fillerData;
  } else {
  HIP_CHECK(hipFree(fillerData));
  }
}

template <typename T> static inline T* createFillerData(size_t count, size_t value, bool fromHost) {
  if (fromHost) {
    T* fillerData = new T[count];
    std::fill(fillerData, fillerData + count, value);
    return fillerData;
  } else {
    T* fillerData;
    HIP_CHECK(hipMalloc(&fillerData, count * sizeof(T)));
    HIP_CHECK(hipMemset(fillerData, value, count * sizeof(T)));
    return fillerData;
  }
}

static void checkForSync(hipStream_t stream, bool async, allocType type, bool fromHost) {
  if (fromHost) {
    if (async && type == allocType::deviceMalloc) {
      HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
    } else {
      REQUIRE(true);
    }
  } else {
    HIP_CHECK(hipStreamQuery(stream));
  }
}


// Helper function to run tests for hipMemset allocation types
template <typename T>
static void runMemcpyTests(hipStream_t stream, bool async, allocType type, memType memType,
                           MultiDData data) {
  bool fromHost = GENERATE(true, false);

  std::pair<T*, T*> aPtr = initMemory<T>(type, memType, data);
  size_t sizeInBytes = data.getCount();

  // filler data for device memory created beforehand as it uses memset
  // which might interfere with synchronization testing
  auto fillerData = createFillerData<T>(sizeInBytes, testValue, fromHost);
  CAPTURE(type, memType, data.width, data.height, data.depth, stream, async, fromHost, sizeInBytes);

  launchLongRunningKernel(100, stream);

  memcpyCheck(type, memType, aPtr.first, data, fillerData, async, stream, fromHost);
  checkForSync(stream, async, type, fromHost);
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
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-127 - Sync behaviour differs on AMD and Nvidia");
  return;
#endif
  allocType type = GENERATE(allocType::deviceMalloc, allocType::hostMalloc, allocType::hostRegisted,
                            allocType::devRegistered);
  memType memcpy_type = memType::hipMem;
  MultiDData data;
  data.width = 1;

  doMemTest<char>(runMemcpyTests<char>, type, memcpy_type, data);
}

TEST_CASE("Unit_hipMemcpy2DSync") {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);

  memType memcpy_type = memType::hipMem2D;
  MultiDData data;
  data.width = 1;
  data.height = 1;

  doMemTest<char>(runMemcpyTests<char>, mallocType, memcpy_type, data);
}

TEST_CASE("Unit_hipMemcpy3DSync") {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);

  memType memcpy_type = memType::hipMem3D;
  MultiDData data;
  data.width = 1;
  data.height = 1;
  data.depth = 1;

  doMemTest<char>(runMemcpyTests<char>, mallocType, memcpy_type, data);
}
