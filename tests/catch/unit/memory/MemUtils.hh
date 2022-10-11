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


namespace mem_utils {

enum class allocType { deviceMalloc, hostMalloc, hostRegisted, devRegistered };
enum class memType { hipMem, hipMemsetD8, hipMemsetD16, hipMemsetD32, hipMem2D, hipMem3D };

// helper struct containing vars needed for 2D and 3D mem Testing
struct MultiDData {
  size_t width{};  // in elements not bytes
  // set to 0 for 1D
  size_t height{};                                     // in elements not bytes
  size_t getH() { return height == 0 ? 1 : height; };  // return 1 if height == 0 || height
  // set to 0 for 2D
  size_t depth{};                                    // in elements not bytes
  size_t getD() { return depth == 0 ? 1 : depth; };  // return 1 if depth == 0 || depth
  size_t pitch{};                                    // pitch = (width * sizeofData) + alignment
  size_t offset{};  // for simplicity use same offset for x,y and z dimentions of memory
  size_t getCount() { return width * getH() * getD(); }
};

// set of helper functions to tidy the nested switch statements
template <typename T>
static inline std::pair<T*, T*> deviceMallocHelper(memType memType, size_t dataW, size_t dataH,
                                                   size_t dataD, size_t& dataPitch) {
  constexpr size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr{};
  switch (memType) {
    case memType::hipMem3D: {
      hipPitchedPtr pitchedAPtr;
      hipExtent extent = make_hipExtent(dataW * elementSize, dataH, dataD);

      HIP_CHECK(hipMalloc3D(&pitchedAPtr, extent));
      aPtr = reinterpret_cast<T*>(pitchedAPtr.ptr);
      dataPitch = pitchedAPtr.pitch;
      break;
    }

    case memType::hipMem2D:
      HIP_CHECK(
          hipMallocPitch(reinterpret_cast<void**>(&aPtr), &dataPitch, dataW * elementSize, dataH));

      break;

    default:
      HIP_CHECK(hipMalloc(&aPtr, sizeInBytes));
      dataPitch = dataW * elementSize;
      break;
  }
  return {aPtr, nullptr};
}

template <typename T>
static inline std::pair<T*, T*> hostMallocHelper(size_t dataW, size_t dataH, size_t dataD,
                                                 size_t& dataPitch) {
  constexpr size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr;

  HIP_CHECK(hipHostMalloc(&aPtr, sizeInBytes));
  dataPitch = dataW * elementSize;

  return {aPtr, nullptr};
}

template <typename T>
static inline std::pair<T*, T*> hostRegisteredHelper(size_t dataW, size_t dataH, size_t dataD,
                                                     size_t& dataPitch) {
  constexpr size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr = new T[dataW * dataH * dataD];

  HIP_CHECK(hipHostRegister(aPtr, sizeInBytes, hipHostRegisterDefault));

  dataPitch = dataW * elementSize;
  return {aPtr, nullptr};
}

template <typename T>
static inline std::pair<T*, T*> devRegisteredHelper(size_t dataW, size_t dataH, size_t dataD,
                                                    size_t& dataPitch) {
  constexpr size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  T* aPtr = new T[dataW * dataH * dataD];
  T* retPtr{};

  HIP_CHECK(hipHostRegister(aPtr, sizeInBytes, hipHostRegisterDefault));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&retPtr), aPtr, 0));

  dataPitch = dataW * elementSize;
  // keep the address of the host memory
  return {retPtr, aPtr};
}

/*
 * helper function to allocate memory and set it to a value.
 * return a pair of pointers due to the device registered allocation case, we need to keep track of
 * the pointer to host memory to be able to unregister and free it
 */
template <typename T>
static inline std::pair<T*, T*> initMemory(allocType type, memType memType, MultiDData& data) {
  std::pair<T*, T*> retPtr{};
  // check different types of allocation
  switch (type) {
    case allocType::deviceMalloc:
      retPtr = deviceMallocHelper<T>(memType, data.width, data.getH(), data.getD(), data.pitch);
      break;

    case allocType::hostMalloc:
      retPtr = hostMallocHelper<T>(data.width, data.getH(), data.getD(), data.pitch);
      break;

    case allocType::hostRegisted:
      retPtr = hostRegisteredHelper<T>(data.width, data.getH(), data.getD(), data.pitch);
      break;

    case allocType::devRegistered:
      retPtr = devRegisteredHelper<T>(data.width, data.getH(), data.getD(), data.pitch);
      break;

    default:
      REQUIRE(false);
      break;
  }
  return retPtr;
}
// create a hipMemcpy3DParams struct for the 3d version of memcpy to verify the memset operation
template <typename T>
hipMemcpy3DParms createParams(hipMemcpyKind kind, T* src, T* host_dst, size_t srcPitch,
                              size_t dataW, size_t dataH, size_t dataD) {
  hipMemcpy3DParms p = {};
  p.kind = kind;

  p.srcPtr.ptr = src;
  p.srcPtr.pitch = srcPitch;
  p.srcPtr.xsize = dataW;
  p.srcPtr.ysize = dataH;

  p.dstPtr.ptr = host_dst;
  p.dstPtr.pitch = dataW * sizeof(T);
  p.dstPtr.xsize = dataW;
  p.dstPtr.ysize = dataH;

  hipExtent extent = make_hipExtent(dataW * sizeof(T), dataH, dataD);
  p.extent = extent;

  return p;
}

// set of helper functions to tidy the nested switch statements
template <typename T>
static inline void deviceMallocCopy(memType memType, T* aPtr, T* hostMem, size_t dataW,
                                    size_t dataH, size_t dataD, size_t& dataPitch) {
  constexpr size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  switch (memType) {
    case memType::hipMem3D: {
      hipMemcpy3DParms params =
          createParams(hipMemcpyDeviceToHost, aPtr, hostMem, dataPitch, dataW, dataH, dataD);
      HIP_CHECK(hipMemcpy3D(&params));
      break;
    }

    case memType::hipMem2D:
      HIP_CHECK(hipMemcpy2D(hostMem, dataW * elementSize, aPtr, dataPitch, dataW, dataH,
                            hipMemcpyDeviceToHost));
      break;

    default:
      HIP_CHECK(hipMemcpy(hostMem, aPtr, sizeInBytes, hipMemcpyDeviceToHost));
      break;
  }
}

template <typename T>
static inline void hostCopy(memType memType, T* aPtr, T* hostMem, size_t dataW, size_t dataH,
                            size_t dataD, size_t& dataPitch) {
  constexpr size_t elementSize = sizeof(T);
  size_t sizeInBytes = elementSize * dataW * dataH * dataD;
  switch (memType) {
    case memType::hipMem3D: {
      hipMemcpy3DParms params =
          createParams(hipMemcpyHostToHost, aPtr, hostMem, dataPitch, dataW, dataH, dataD);

      HIP_CHECK(hipMemcpy3D(&params));
      break;
    }

    case memType::hipMem2D:
      HIP_CHECK(hipMemcpy2D(hostMem, dataW * elementSize, aPtr, dataPitch, dataW, dataH,
                            hipMemcpyHostToHost));
      break;

    default:
      HIP_CHECK(hipMemcpy(hostMem, aPtr, sizeInBytes, hipMemcpyHostToHost));
      break;
  }
}

template <typename T>
static inline void devRegisteredCopy(memType memType, T* aPtr, T* hostMem, size_t dataW,
                                     size_t dataH, size_t dataD, size_t& dataPitch) {
  constexpr size_t elementSize = sizeof(T);

  switch (memType) {
    case memType::hipMem3D: {
      hipMemcpy3DParms params =
          createParams(hipMemcpyDeviceToHost, aPtr, hostMem, dataPitch, dataW, dataH, dataD);

      HIP_CHECK(hipMemcpy3D(&params));
      break;
    }

    case memType::hipMem2D:
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
  } else if (data.depth == 0) {
    return (data.offset + (data.pitch * data.offset));
  } else {  // 2D or 3D
    return (data.offset + (data.pitch * data.offset) + (data.pitch * data.offset * data.height));
  }
}

/*
 * Function to allow reuse of functions for testing versions of the memset API, at a specified
 * offset
 */
template <typename T>
static inline void memsetCheck(T* aPtr, size_t value, memType memType, MultiDData& data,
                               hipStream_t stream = nullptr, bool async = true) {
  size_t count = data.getCount();
  size_t ptrOffset{};
  switch (memType) {
    case memType::hipMem:
      if (async) {
        HIP_CHECK(hipMemsetAsync(aPtr + data.offset, value, count * sizeof(T), stream));
      } else {
        HIP_CHECK(hipMemset(aPtr + data.offset, value, count * sizeof(T)));
      }
      break;

    case memType::hipMemsetD8:
      if (async) {
        HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value,
                                   count, stream));
      } else {
        HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value, count));
      }
      break;

    case memType::hipMemsetD16:
      if (async) {
        HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value,
                                    count, stream));
      } else {
        HIP_CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value, count));
      }
      break;

    case memType::hipMemsetD32:
      if (async) {
        HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value,
                                    count, stream));
      } else {
        HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(aPtr + data.offset), value, count));
      }
      break;

    case memType::hipMem2D:
      ptrOffset = getPtrOffset(data);
      if (async) {
        HIP_CHECK(
            hipMemset2DAsync(aPtr + ptrOffset, data.pitch, value, data.width, data.height, stream));
      } else {
        HIP_CHECK(hipMemset2D(aPtr + ptrOffset, data.pitch, value, data.width, data.height));
      }
      break;

    case memType::hipMem3D: {
      ptrOffset = getPtrOffset(data);
      hipExtent extent = make_hipExtent(data.width * sizeof(T), data.height, data.depth);

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
    }
    default:
      REQUIRE(false);
      break;
  }
}

template <typename T> static inline void freeStuff(T* aPtr, allocType type) {
  switch (type) {
    case allocType::deviceMalloc:
      HIP_CHECK(hipFree(aPtr));
      break;
    case allocType::hostMalloc:
      HIP_CHECK(hipHostFree(aPtr));
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
                              memType memType) {
  std::unique_ptr<T[]> hostPtr = std::make_unique<T[]>(data.getCount());
  switch (type) {
    case allocType::deviceMalloc:
      deviceMallocCopy(memType, aPtr + getPtrOffset(data), hostPtr.get(), data.width, data.getH(),
                       data.getD(), data.pitch);
      break;
    case allocType::devRegistered:
      devRegisteredCopy(memType, aPtr + getPtrOffset(data), hostPtr.get(), data.width, data.getH(),
                        data.getD(), data.pitch);
      break;
    default:  // host malloc and host registered
      hostCopy(memType, aPtr + getPtrOffset(data), hostPtr.get(), data.width, data.getH(),
               data.getD(), data.pitch);
      break;
  }

  size_t idx;
  bool allMatch{true};
  for (size_t k = 0; k < data.getD(); k++) {
    for (size_t j = 0; j < data.getH(); j++) {
      for (size_t i = 0; i < data.width; i++) {
        idx = data.width * data.getH() * k + data.width * j + i;
        allMatch = allMatch && static_cast<size_t>(hostPtr.get()[idx]) == value;
        if (!allMatch) REQUIRE(false);
      }
    }
  }
}

// function used to abstract the test
template <typename T, typename F, typename... fArgs>
static inline void doMemTest(F func, fArgs... funcArgs) {
  SECTION("Synchronous") { func(nullptr, false, funcArgs...); }
  SECTION("Asynchronous - null stream") { func(nullptr, true, funcArgs...); }
  SECTION("Asynchronous - created stream") {
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    func(stream, true, funcArgs...);
    HIP_CHECK(hipStreamDestroy(stream));
  }
}
}  // namespace mem_utils
