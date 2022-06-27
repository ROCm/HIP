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

#pragma once
#include <memory>
#include <hip_test_common.hh>

#include <hip_test_context.hh>

class DriverContext {
 private:
  hipCtx_t ctx;
  hipDevice_t device;

 public:
  DriverContext();
  ~DriverContext();

  // Rule of three
  DriverContext(const DriverContext& other) = delete;
  DriverContext(DriverContext&& other) noexcept = delete;
};

namespace memset_utils {

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
  size_t depth{};   // in elements not bytes
  size_t pitch{};   // pitch = (width * sizeofData) + alignment
  size_t offset{};  // for simplicity use same offset for x,y and z dimentions of memory
};

// set of helper functions to tidy the nested switch statements
template <typename T>
static inline std::pair<T*, T*> deviceMallocHelper(memSetType memType, size_t dataW, size_t dataH,
                                                  size_t dataD, size_t& dataPitch) {
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
static inline std::pair<T*, T*> hostMallocHelper(size_t dataW, size_t dataH, size_t dataD,
                                                size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr;

  HIP_CHECK(hipHostMalloc(&aPtr, sizeInBytes));
  dataPitch = dataW * elementSize;

  return std::make_pair(aPtr, nullptr);
}

template <typename T>
static inline std::pair<T*, T*> hostRegisteredHelper(size_t dataW, size_t dataH, size_t dataD,
                                                    size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr = new T[dataW * dataH * dataD];

  HIP_CHECK(hipHostRegister(aPtr, sizeInBytes, hipHostRegisterDefault));

  dataPitch = dataW * elementSize;
  return std::make_pair(aPtr, nullptr);
}

template <typename T>
static inline std::pair<T*, T*> devRegisteredHelper(size_t dataW, size_t dataH, size_t dataD,
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

/*
 * helper function to allocate memory and set it to a value.
 * return a pair of pointers due to the device registered allocation case, we need to keep track of
 * the pointer to host memory to be able to unregister and free it
 */
template <typename T>
static inline std::pair<T*, T*> initMemory(allocType type, memSetType memType, MultiDData& data) {
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
static inline void deviceMallocCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW,
                                   size_t dataH, size_t dataD, size_t& dataPitch) {
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
static inline void hostCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW, size_t dataH,
                           size_t dataD, size_t& dataPitch) {
  size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
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
                            hipMemcpyHostToHost));
      break;

    default:
      HIP_CHECK(hipMemcpy(hostMem, aPtr, sizeInBytes, hipMemcpyHostToHost));
      break;
  }
}

template <typename T>
static inline void devRegisteredCopy(memSetType memType, T* aPtr, T* hostMem, size_t dataW,
                                    size_t dataH, size_t dataD, size_t& dataPitch) {
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

/*
 * function returns an offset location in memory based on the provided data, taking pitch into
 * account
 * (for 1D requires data.depth & data.height = 0, for 2D data.depth = 0)
 */
static inline size_t getPtrOffset(MultiDData data) {
  if (data.height == 0) {  // 1D
    return data.offset;
  } else {  // 2D or 3D
    return (data.offset + (data.pitch * data.offset) + (data.pitch * data.offset * data.height));
  }
}


/*
 * Function to allow reuse of functions for testing versions of the memset API, at a specified
 * offset
 */
template <typename T>
static inline void memsetCheck(T* aPtr, size_t value, memSetType memsetType, MultiDData& data,
                              hipStream_t stream = nullptr, bool async = true) {
  size_t dataW = data.width;
  size_t dataH = data.height == 0 ? 1 : data.height;
  size_t dataD = data.depth == 0 ? 1 : data.depth;
  size_t count = dataW * dataH * dataD;
  size_t ptrOffset = getPtrOffset(data);
  switch (memsetType) {
    case memSetType::hipMemset:
      if (async) {
        HIP_CHECK(hipMemsetAsync(aPtr + data.offset, value, count * sizeof(T), stream));
      } else {
        HIP_CHECK(hipMemset(aPtr + data.offset, value, count * sizeof(T)));
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
template <typename T> static inline void freeStuff(T* aPtr, allocType type) {
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

/*
 * Copies device data to host and checks that each element is equal to the
 * specified value
 */
template <typename T>
static inline void verifyData(T* aPtr, size_t value, MultiDData& data, allocType type,
                              memSetType memType) {
  auto dataH = data.height == 0 ? 1 : data.height;
  auto dataD = data.depth == 0 ? 1 : data.depth;
  size_t sizeInBytes = data.pitch * dataH * dataD;
  std::unique_ptr<T[]> hostPtr = std::make_unique<T[]>(data.pitch * dataH * dataD);
  switch (type) {
    case allocType::deviceMalloc:
      deviceMallocCopy(memType, aPtr + getPtrOffset(data), hostPtr.get(), data.width, dataH, dataD,
                       data.pitch);
      break;
    case allocType::hostMalloc:
      hostCopy(memType, aPtr + getPtrOffset(data), hostPtr.get(), data.width, dataH, dataD,
               data.pitch);
      break;
    case allocType::hostRegisted:
      hostCopy(memType, aPtr + getPtrOffset(data), hostPtr.get(), data.width, dataH, dataD,
               data.pitch);
      break;
    case allocType::devRegistered:
      devRegisteredCopy(memType, aPtr + getPtrOffset(data), hostPtr.get(), data.width, dataH, dataD,
                        data.pitch);
      break;
    default:
      break;
  }

  size_t idx;
  bool allMatch{true};
  for (size_t k = 0; k < dataD; k++) {
    for (size_t j = 0; j < dataH; j++) {
      for (size_t i = 0; i < data.width; i++) {
        idx = data.pitch * dataH * k + data.pitch * j + i;
        CAPTURE(sizeInBytes, i, j, k, value, data.pitch, reinterpret_cast<long>(aPtr), type,
                memType);
        allMatch = allMatch && static_cast<size_t>(hostPtr.get()[idx]) == value;
        if (!allMatch) REQUIRE(false);
      }
    }
  }
}

// function used to abstract the test
template <typename T, typename F, typename... fArgs>
static inline void doMemsetTest(F func, fArgs... funcArgs) {
  enum StreamType { NULLSTR, CREATEDSTR };
  auto streamType = GENERATE(NULLSTR, CREATEDSTR);
  hipStream_t stream{nullptr};

  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamCreate(&stream));
  func(stream, funcArgs...);
  if (streamType == CREATEDSTR) HIP_CHECK(hipStreamDestroy(stream));
}


}  // namespace memset_utils
