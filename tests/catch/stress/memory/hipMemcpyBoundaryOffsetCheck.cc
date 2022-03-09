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
/*
This testcase verifies following scenarios
3. Boundary checks with different sizes
5. device offset scenario
*/
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include "sys/types.h"
#include "sys/sysinfo.h"
#endif
static constexpr auto NUM_ELM{4*1024 * 1024};
template<typename T>
class DeviceMemory {
 public:
    explicit DeviceMemory(size_t numElements);
    DeviceMemory() = delete;
    ~DeviceMemory();
    T* A_d() const { return _A_d + _offset; }
    T* B_d() const { return _B_d + _offset; }
    T* C_d() const { return _C_d + _offset; }
    T* C_dd() const { return _C_dd + _offset; }
    size_t maxNumElements() const { return _maxNumElements; }
    void offset(int offset) { _offset = offset; }
    int offset() const { return _offset; }
 private:
    T* _A_d;
    T* _B_d;
    T* _C_d;
    T* _C_dd;
    size_t _maxNumElements;
    int _offset;
};
template <typename T>
DeviceMemory<T>::DeviceMemory(size_t numElements) :
                 _maxNumElements(numElements), _offset(0) {
  T** np = nullptr;
  HipTest::initArrays(&_A_d, &_B_d, &_C_d, np, np, np, numElements, 0);
  size_t sizeElements = numElements * sizeof(T);
  HIP_CHECK(hipMalloc(&_C_dd, sizeElements));
}
template <typename T>
DeviceMemory<T>::~DeviceMemory() {
  T* np = nullptr;
  HipTest::freeArrays<T>(_A_d, _B_d, _C_d, np, np, np, 0);
  HIP_CHECK(hipFree(_C_dd));
  _C_dd = NULL;
}
template <typename T>
class HostMemory {
 public:
    HostMemory(size_t numElements, bool usePinnedHost);
    HostMemory() = delete;
    void reset(size_t numElements, bool full = false);
    ~HostMemory();
    T* A_h() const { return _A_h + _offset; }
    T* B_h() const { return _B_h + _offset; }
    T* C_h() const { return _C_h + _offset; }
    size_t maxNumElements() const { return _maxNumElements; }
    void offset(int offset) { _offset = offset; }
    int offset() const { return _offset; }
    // Host arrays, secondary copy
    T* A_hh;
    T* B_hh;
    bool _usePinnedHost;
 private:
    size_t _maxNumElements;
    int _offset;
    // Host arrays
    T* _A_h;
    T* _B_h;
    T* _C_h;
};
  template <typename T>
HostMemory<T>::HostMemory(size_t numElements, bool usePinnedHost)
  : _usePinnedHost(usePinnedHost), _maxNumElements(numElements), _offset(0) {
    T** np = nullptr;
    HipTest::initArrays(np, np, np, &_A_h, &_B_h, &_C_h,
                        numElements, usePinnedHost);
    A_hh = NULL;
    B_hh = NULL;
    size_t sizeElements = numElements * sizeof(T);
    if (usePinnedHost) {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_hh), sizeElements,
                              hipHostMallocDefault));
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&B_hh), sizeElements,
                              hipHostMallocDefault));
    } else {
      A_hh = reinterpret_cast<T*>(malloc(sizeElements));
      B_hh = reinterpret_cast<T*>(malloc(sizeElements));
    }
  }
template <typename T>
void HostMemory<T>::reset(size_t numElements, bool full) {
  // Initialize the host data:
  for (size_t i = 0; i < numElements; i++) {
    (A_hh)[i] = 1097.0 + i;
    (B_hh)[i] = 1492.0 + i;  // Phi
    if (full) {
      (_A_h)[i] = 3.146f + i;  // Pi
      (_B_h)[i] = 1.618f + i;  // Phi
    }
  }
}
template <typename T>
HostMemory<T>::~HostMemory() {
  HipTest::freeArraysForHost(_A_h, _B_h, _C_h, _usePinnedHost);
  if (_usePinnedHost) {
    HIP_CHECK(hipHostFree(A_hh));
    HIP_CHECK(hipHostFree(B_hh));
  } else {
    free(A_hh);
    free(B_hh);
  }
}
#ifdef _WIN32
void memcpytest2_get_host_memory(size_t *free, size_t *total) {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  // Windows doesn't allow allocating more than half of system memory to the gpu
  // Since the runtime also needs space for its internal allocations,
  // we should not try to allocate more than 40% of reported system memory,
  // otherwise we can run into OOM issues.
  *free = static_cast<size_t>(0.4 * status.ullAvailPhys);
  *total = static_cast<size_t>(0.4 * status.ullTotalPhys);
}
#else
struct sysinfo memInfo;
void memcpytest2_get_host_memory(size_t  *free, size_t *total) {
  sysinfo(&memInfo);
  uint64_t freePhysMem = memInfo.freeram;
  freePhysMem *= memInfo.mem_unit;
  *free = freePhysMem;
  uint64_t totalPhysMem = memInfo.totalram;
  totalPhysMem *= memInfo.mem_unit;
  *total = totalPhysMem;
}
#endif
//---
// Test many different kinds of memory copies.
// The subroutine allocates memory , copies to device, runs a vector
//  add kernel, copies back, and
// checks the result.
//
// IN: numElements  controls the number of elements used for allocations.
// IN: usePinnedHost : If true, allocate host with hipHostMalloc and is pinned
// else allocate host
// memory with malloc. IN: useHostToHost : If true, add an extra
// host-to-host copy. IN:
// useDeviceToDevice : If true, add an extra deviceto-device copy after
// result is produced. IN:
// useMemkindDefault : If true, use memkinddefault
// (runtime figures out direction).  if false, use
// explicit memcpy direction.
//
template <typename T>
void memcpytest2(DeviceMemory<T>* dmem, HostMemory<T>* hmem,
    size_t numElements, bool useHostToHost,
    bool useDeviceToDevice, bool useMemkindDefault) {
  size_t sizeElements = numElements * sizeof(T);
  hmem->reset(numElements);
  assert(numElements <= dmem->maxNumElements());
  assert(numElements <= hmem->maxNumElements());
  if (useHostToHost) {
    // Do some extra host-to-host copies here to mix things up:
    HIP_CHECK(hipMemcpy(hmem->A_hh, hmem->A_h(), sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToHost));
    HIP_CHECK(hipMemcpy(hmem->B_hh, hmem->B_h(), sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToHost));
    HIP_CHECK(hipMemcpy(dmem->A_d(), hmem->A_hh, sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dmem->B_d(), hmem->B_hh, sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
  } else {
    HIP_CHECK(hipMemcpy(dmem->A_d(), hmem->A_h(), sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dmem->B_d(), hmem->B_h(), sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
  }
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(1), dim3(1), 0, 0,
      static_cast<const T*>(dmem->A_d()), static_cast<const T*>(dmem->B_d()),
      dmem->C_d(), numElements);
  if (useDeviceToDevice) {
    // Do an extra device-to-device copy here to mix things up:
    HIP_CHECK(hipMemcpy(dmem->C_dd(), dmem->C_d(), sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyDeviceToDevice));
    // Destroy the original dmem->C_d():
    HIP_CHECK(hipMemset(dmem->C_d(), 0x5A, sizeElements));
    HIP_CHECK(hipMemcpy(hmem->C_h(), dmem->C_dd(), sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyDeviceToHost));
  } else {
    HIP_CHECK(hipMemcpy(hmem->C_h(), dmem->C_d(), sizeElements,
          useMemkindDefault ? hipMemcpyDefault : hipMemcpyDeviceToHost));
  }
  HIP_CHECK(hipDeviceSynchronize());
  HipTest::checkVectorADD(hmem->A_h(), hmem->B_h(), hmem->C_h(), numElements);
}
// Try all the 16 possible combinations to memcpytest2 - usePinnedHost,
// useHostToHost,
// useDeviceToDevice, useMemkindDefault
template <typename T>
void memcpytest2_for_type(size_t numElements) {
  DeviceMemory<T> memD(numElements);
  HostMemory<T> memU(numElements, 0 /*usePinnedHost*/);
  HostMemory<T> memP(numElements, 1 /*usePinnedHost*/);
  for (int usePinnedHost = 0; usePinnedHost <= 1; usePinnedHost++) {
    for (int useHostToHost = 0; useHostToHost <= 1; useHostToHost++) {
      for (int useDeviceToDevice = 0; useDeviceToDevice <= 1;
          useDeviceToDevice++) {
        for (int useMemkindDefault = 0; useMemkindDefault <= 1;
            useMemkindDefault++) {
          memcpytest2<T>(&memD, usePinnedHost ? &memP : &memU,
              numElements, useHostToHost,
              useDeviceToDevice, useMemkindDefault);
        }
      }
    }
  }
}
// Try many different sizes to memory copy.
template <typename T>
void memcpytest2_sizes(size_t maxElem = 0) {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  size_t free, total, freeCPU, totalCPU;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  memcpytest2_get_host_memory(&freeCPU, &totalCPU);
  if (maxElem == 0) {
    // Use lesser maxElem if not enough host memory available
    size_t maxElemGPU = free / sizeof(T) / 8;
    size_t maxElemCPU = freeCPU / sizeof(T) / 8;
    maxElem = maxElemGPU < maxElemCPU ? maxElemGPU : maxElemCPU;
  }
  HIP_CHECK(hipDeviceReset());
  DeviceMemory<T> memD(maxElem);
  HostMemory<T> memU(maxElem, 0 /*usePinnedHost*/);
  HostMemory<T> memP(maxElem, 1 /*usePinnedHost*/);
  for (size_t elem = 1; elem <= maxElem; elem *= 2) {
    memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
    memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
  }
}
// Try many different sizes to memory copy.
template <typename T>
void memcpytest2_offsets(size_t maxElem, bool devOffsets, bool hostOffsets) {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  size_t free, total;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  HIP_CHECK(hipDeviceReset());
  DeviceMemory<T> memD(maxElem);
  HostMemory<T> memU(maxElem, 0 /*usePinnedHost*/);
  HostMemory<T> memP(maxElem, 1 /*usePinnedHost*/);
  size_t elem = maxElem / 2;
  for (size_t offset = 0; offset < 512; offset++) {
    assert(elem + offset < maxElem);
    if (devOffsets) {
      memD.offset(offset);
    }
    if (hostOffsets) {
      memU.offset(offset);
      memP.offset(offset);
    }
    memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
    memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
  }
  for (size_t offset = 512; offset < elem; offset *= 2) {
    assert(elem + offset < maxElem);
    if (devOffsets) {
      memD.offset(offset);
    }
    if (hostOffsets) {
      memU.offset(offset);
      memP.offset(offset);
    }
    memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
    memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
  }
}
// Create multiple threads to stress multi-thread locking behavior in the
// allocation/deallocation/tracking logic:
template <typename T>
void multiThread_1(bool serialize, bool usePinnedHost) {
  DeviceMemory<T> memD(NUM_ELM);
  HostMemory<T> mem1(NUM_ELM, usePinnedHost);
  HostMemory<T> mem2(NUM_ELM, usePinnedHost);
  std::thread t1(memcpytest2<T>, &memD, &mem1, NUM_ELM, 0, 0, 0);
  if (serialize) {
    t1.join();
  }
  std::thread t2(memcpytest2<T>, &memD, &mem2, NUM_ELM, 0, 0, 0);
  if (serialize) {
    t2.join();
  }
}
/*
This testcase verfies the boundary checks of hipMemcpy API for different sizes
*/
TEST_CASE("Unit_hipMemcpy_BoundaryCheck") {
  size_t maxElem = 32 * 1024 * 1024;
  DeviceMemory<float> memD(maxElem);
  HostMemory<float> memU(maxElem, 0 /*usePinnedHost*/);
  HostMemory<float> memP(maxElem, 0 /*usePinnedHost*/);
  memcpytest2<float>(&memD, &memU, 32 * 1024 * 1024, 0, 0, 0);
  auto sizes = GENERATE(15 * 1024 * 1024, 16 * 1024 * 1024,
                        16 * 1024 * 1024 + 16 * 1024,
                        16 * 1024 * 1024 + 512 * 1024,
                        17 * 1024 * 1024 + 1024,
                        32 * 1024 * 1024);
  memcpytest2<float>(&memD, &memP, sizes, 0, 0, 0);
}

/*
This testcase verifies the device offsets
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpy_DeviceOffsets", "", float, double) {
  HIP_CHECK(hipDeviceReset());
  size_t maxSize = 256 * 1024;
  memcpytest2_offsets<TestType>(maxSize, true, false);
  memcpytest2_offsets<TestType>(maxSize, false, true);
}
