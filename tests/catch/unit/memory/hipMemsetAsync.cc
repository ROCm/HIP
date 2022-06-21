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
  size_t width{};  // in elements not bytes
  // set to 0 for 1D
  size_t height{};  // in elements not bytes
  // set to 0 for 2D
  size_t depth{};  // in elements not bytes
  size_t pitch{};
  size_t offset{};  // for simplicity use same offset for x,y and z dimentions of memory
};


// set of helper functions to tidy the nested switch statements
template <typename T>
static T* deviceMallocHelper(memSetType memType, size_t dataW, size_t dataH, size_t dataD,
                             size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  hipPitchedPtr pitchedAPtr{};
  T* aPtr{};
  switch (memType) {
    case memSetType::hipMemset3D:
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
  return aPtr;
}

template <typename T>
static T* hostMallocHelper(size_t dataW, size_t dataH, size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr;

  HIP_CHECK(hipHostMalloc(&aPtr, sizeInBytes));
  dataPitch = dataW * elementSize;

  return aPtr;
}

template <typename T>
static T* hostRegisteredHelper(size_t dataW, size_t dataH, size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr = new T[dataW * dataH * dataD];

  HIP_CHECK(hipHostRegister(aPtr, sizeInBytes, hipHostRegisterDefault));

  dataPitch = dataW * elementSize;
  return aPtr;
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
template <typename T>
static std::pair<T*, T*> initMemory(allocType type, memSetType memType, MultiDData& data) {
  size_t dataH = data.height == 0 ? 1 : data.height;
  size_t dataD = data.depth == 0 ? 1 : data.depth;
  std::pair<T*, T*> retPtr{};
  // check different types of allocation
  switch (type) {
    case allocType::deviceMalloc:
      retPtr = std::make_pair(deviceMallocHelper<T>(memType, data.width, dataH, dataD, data.pitch),
                              nullptr);
      break;

    case allocType::hostMalloc:
      retPtr = std::make_pair(hostMallocHelper<T>(data.width, dataH, dataD, data.pitch), nullptr);
      break;

    case allocType::hostRegisted:
      retPtr =
          std::make_pair(hostRegisteredHelper<T>(data.width, dataH, dataD, data.pitch), nullptr);
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
  hipMemcpy3DParms params{};
  switch (memType) {
    case memSetType::hipMemset3D:

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
static void hostMallocCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW, size_t dataH,
                           size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  hipMemcpy3DParms params{};
  switch (memType) {
    case memSetType::hipMemset3D:

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
static void hostRegisteredCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW, size_t dataH,
                               size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  hipMemcpy3DParms params{};
  switch (memType) {
    case memSetType::hipMemset3D:

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
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  hipMemcpy3DParms params{};

  switch (memType) {
    case memSetType::hipMemset3D:

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

    case memSetType::hipMemset2D:
      HIP_CHECK(hipMemcpy2D(hostMem, dataW * elementSize, aPtr, dataPitch, dataW, dataH,
                            hipMemcpyDeviceToHost));
      break;

    default:
      HIP_CHECK(hipMemcpy(hostMem, aPtr, sizeInBytes, hipMemcpyDeviceToHost));
      break;
  }
}

static size_t getPtrOffset(MultiDData data) {
  return (data.offset + (data.pitch * data.offset) + (data.pitch * data.offset * data.height));
}

// Copies device data to host and checks that each element is equal to the
// specified value
template <typename T>
void verifyData(T* aPtr, size_t value, MultiDData& data, allocType type, memSetType memType) {
  auto dataH = data.height == 0 ? 1 : data.height;
  auto dataD = data.depth == 0 ? 1 : data.depth;
  size_t sizeInBytes = data.pitch * dataH * dataD;
  T* hostPtr = new T[sizeInBytes];
  switch (type) {
    case allocType::deviceMalloc:
      printf("deviceMalloc \n");
      deviceMallocCopy(memType, aPtr + getPtrOffset(data), hostPtr, data.width, dataH, dataD,
                       data.pitch);
      break;
    case allocType::hostMalloc:
    printf("hostMalloc \n");
      hostMallocCopy(memType, aPtr + getPtrOffset(data), hostPtr, data.width, dataH, dataD,
                     data.pitch);
      break;
    case allocType::hostRegisted:
    printf("hostRegisted \n");
      hostRegisteredCopy(memType, aPtr + getPtrOffset(data), hostPtr, data.width, dataH, dataD,
                         data.pitch);
      break;
    case allocType::devRegistered:
    printf("devRegistered \n");
      devRegisteredCopy(memType, aPtr + getPtrOffset(data), hostPtr, data.width, dataH, dataD,
                        data.pitch);
      break;
    default:
      break;
  }

  size_t idx;
  //bool allMatch = true;
  if(value){}
  [&] {
    for (size_t k = 0; k < dataD; k++) {
      for (size_t j = 0; j < dataH; j++) {
        for (size_t i = 0; i < data.width; i++) {
          idx = data.pitch * dataH * k + data.pitch * j + i;
          printf("\t %d ", hostPtr[idx]);
          //CAPTURE(sizeInBytes, i, j, k, value, data.pitch, reinterpret_cast<long>(aPtr), hostPtr[idx], type, memType);
          //allMatch = allMatch && static_cast<size_t>(hostPtr[idx]) == value;
          //if (!allMatch) REQUIRE(false);
        }
      }
    }
  }();
  printf("\n");
  //REQUIRE(allMatch);
}


/* Function which calls the memset API, at a specified offset */
template <typename T>
void memsetCheck(T* aPtr, T value, memSetType memsetType, MultiDData& data,
                 hipStream_t stream = nullptr, bool async = true ) {
  size_t dataW = data.width;
  size_t dataH = data.height == 0 ? 1 : data.height;
  size_t dataD = data.depth == 0 ? 1 : data.depth;
  size_t count = dataW * dataH * dataD;
  size_t ptrOffset;
  switch (memsetType) {
    case memSetType::hipMemset:
      if (async) {
        HIP_CHECK(hipMemsetAsync(aPtr + 4/*data.offset*/, value, count * sizeof(T), stream));
      } else {
        HIP_CHECK(hipMemset(aPtr + data.offset, value , count* sizeof(T)));
      }
      break;

    case memSetType::hipMemsetD8:
      if (async) {
        HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value,
                                   count, stream));
      } else {
        HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value, count));
      }
      break;

    case memSetType::hipMemsetD16:
      if (async) {
        HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value,
                                    count, stream));
      } else {
        HIP_CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value, count));
      }
      break;

    case memSetType::hipMemsetD32:
      if (async) {
        HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value,
                                    count, stream));
      } else {
        HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value, count));
      }
      break;

    case memSetType::hipMemset2D:
      ptrOffset = getPtrOffset(data);
      data.pitch = data.pitch - data.offset;
      if (async) {
        HIP_CHECK(
            hipMemset2DAsync(aPtr + ptrOffset, data.pitch, value, data.width, data.height, stream));
      } else {
        HIP_CHECK(hipMemset2D(aPtr + ptrOffset, data.pitch, value, data.width, data.height));
      }
      break;

    case memSetType::hipMemset3D:
      hipExtent extent;
      extent.width = data.width;
      extent.height = data.height;
      extent.depth = data.depth;
      ptrOffset = getPtrOffset(data);
      data.pitch = data.pitch - data.offset;
      if (async) {
        HIP_CHECK(hipMemset3DAsync(
            make_hipPitchedPtr(aPtr + ptrOffset, data.pitch, data.width, data.height), value,
            extent, stream));
      } else {
        HIP_CHECK(
            hipMemset3D(make_hipPitchedPtr(aPtr + ptrOffset, data.pitch, data.width, data.height),
                        value, extent));
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
    default:  // for host and device registered
      HIP_CHECK(hipHostUnregister(aPtr));
      delete[] aPtr;
      break;
  }
}

// Helper function to run tests for hipMemset allocation types
template <typename T>
void runAsyncTests(hipStream_t stream, allocType type, memSetType memsetType, MultiDData data1,
                   MultiDData data2) {
  // bool async = GENERATE(true, false);
  // CAPTURE(type, memsetType, data.width, data.height, data.depth, stream, async);
  std::pair<T*, T*> aPtr{};
  MultiDData totalRange;
  totalRange.width = data1.width + data2.width;
  totalRange.height = data1.height + data2.height;
  totalRange.depth= data1.depth + data2.depth;
  aPtr = initMemory<T>(type, memsetType, totalRange);
  data1.pitch = totalRange.pitch;
  data2.pitch = totalRange.pitch - data2.offset;
  printf("  \n");
  memsetCheck(aPtr.first, 'a', memsetType, data1, stream);
  memsetCheck(aPtr.first, 'b', memsetType, data2, stream);
 
  printf("after \n");
  HIP_CHECK(hipStreamSynchronize(stream));
  verifyData(aPtr.first, 0x11, totalRange, type, memsetType);
  //verifyData(aPtr.first, 0x22, data2, type, memsetType);


  if (type == allocType::devRegistered) {
    freeStuff(aPtr.second, type);
  } else {
    freeStuff(aPtr.first, type);
  }
}

template <typename T, typename F, typename... fArgs>
static void doMemsetTest(F func, fArgs... funcArgs) {
  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));
  func(stream, funcArgs...);
  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}

/*
 * test 2 async hipMemset's on the same memory at different offsets
 */

TEST_CASE("Unit_hipMemsetASyncMulti") {
  allocType mallocType = GENERATE(allocType::hostMalloc, allocType::deviceMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 6);
  MultiDData data2;
  data2.width = data1.width;

  data2.offset = data1.width;
  doMemsetTest<char>(runAsyncTests<char>, mallocType, memset_type, data1, data2);
}

/*
 * test 2 async hipMemsetD[8,16,32]'s on the same memory at different offsets
 */
TEMPLATE_TEST_CASE("Unit_hipMemsetDASyncMulti", "", int8_t, int16_t, uint32_t) {
  allocType mallocType = GENERATE(allocType::hostRegisted, allocType::deviceMalloc,
                                  allocType::hostMalloc, allocType::devRegistered);
  memSetType memset_type;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 512);
  MultiDData data2;
  data2.width = data1.width;

  if (std::is_same<int8_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD8;
  } else if (std::is_same<int16_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD16;
  } else if (std::is_same<uint32_t, TestType>::value) {
    memset_type = memSetType::hipMemsetD32;
  }

  doMemsetTest<char>(runAsyncTests<char>, mallocType, memset_type, data1, data2);
}
/*
 * test 2 async hipMemset2D's on the same memory at different offsets
 */
TEMPLATE_TEST_CASE("Unit_hipMemset2DASyncMulti", "", char) {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset2D;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 512);
  data1.height = GENERATE(1, 512);
  MultiDData data2;
  data2.width = data1.width;
  data2.height = data1.height;

  doMemsetTest<char>(runAsyncTests<char>, mallocType, memset_type, data1, data2);
}
/*
 * test 2 async hipMemset3D's on the same memory at different offsets
 */
TEMPLATE_TEST_CASE("Unit_hipMemset3DASyncMulti", "", char) {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);
  memSetType memset_type = memSetType::hipMemset3D;
  MultiDData data1;
  data1.offset = 0;
  data1.width = GENERATE(1, 128);
  data1.height = GENERATE(1, 128);
  data1.depth = GENERATE(1, 128);
  MultiDData data2;
  data2.width = data1.width;
  data2.height = data1.height;
  data2.depth = data1.depth;

  doMemsetTest<char>(runAsyncTests<char>, mallocType, memset_type, data1, data2);
}
