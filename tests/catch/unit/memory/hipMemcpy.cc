/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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
1. hipMemcpy API along with kernel launch with different data types
2. H2D-D2D-D2H scenarios for unpinned and pinned memory
3. Boundary checks with different sizes
4. Multithread scenario
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
static unsigned blocksPerCU{6};  // to hide latency
static unsigned threadsPerBlock{256};

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
  HIP_CHECK(hipGetLastError());

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


  printf("  %s success\n", __func__);
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
This testcase verifies hipMemcpy API
Initializes device variables
Launches kernel and performs the sum of device variables
copies the result to host variable and validates the result.
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpy_KernelLaunch", "", int, float,
                   double) {
  size_t Nbytes = NUM_ELM * sizeof(TestType);

  TestType *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM, false);

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(1), dim3(1), 0, 0,
                     static_cast<const TestType*>(A_d),
                     static_cast<const TestType*>(B_d), C_d, NUM_ELM);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipDeviceSynchronize());
  HipTest::checkVectorADD(A_h, B_h, C_h, NUM_ELM);

  HipTest::freeArrays<TestType>(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/*
This testcase verifies the following scenarios
1. H2H,H2PinMem and PinnedMem2Host
2. H2D-D2D-D2H in same GPU
3. Pinned Host Memory to device variables in same GPU
4. Device context change
5. H2D-D2D-D2H peer GPU
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpy_H2H-H2D-D2H-H2PinMem", "", int,
                   float, double) {
  TestType *A_d{nullptr}, *B_d{nullptr};
  TestType *A_h{nullptr}, *B_h{nullptr};
  TestType *A_Ph{nullptr}, *B_Ph{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<TestType>(&A_d, &B_d, nullptr,
                             &A_h, &B_h, nullptr,
                             NUM_ELM*sizeof(TestType));
  HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                             &A_Ph, &B_Ph, nullptr,
                             NUM_ELM*sizeof(TestType), true);

  SECTION("H2H, H2PinMem and PinMem2H") {
    HIP_CHECK(hipMemcpy(B_h, A_h, NUM_ELM*sizeof(TestType),
                        hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(A_Ph, B_h, NUM_ELM*sizeof(TestType),
                        hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_Ph, A_Ph, NUM_ELM*sizeof(TestType),
                        hipMemcpyDefault));
    HipTest::checkTest(A_h, B_Ph, NUM_ELM);
  }

  SECTION("H2D-D2D-D2H-SameGPU") {
    HIP_CHECK(hipMemcpy(A_d, A_h, NUM_ELM*sizeof(TestType), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_d, A_d, NUM_ELM*sizeof(TestType), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_h, B_d, NUM_ELM*sizeof(TestType), hipMemcpyDefault));
    HipTest::checkTest(A_h, B_h, NUM_ELM);
  }

  SECTION("pH2D-D2D-D2pH-SameGPU") {
    HIP_CHECK(hipMemcpy(A_d, A_Ph, NUM_ELM*sizeof(TestType),
                        hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_d, A_d, NUM_ELM*sizeof(TestType), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_Ph, B_d, NUM_ELM*sizeof(TestType),
                        hipMemcpyDefault));
    HipTest::checkTest(A_Ph, B_Ph, NUM_ELM);
  }
  SECTION("H2D-D2D-D2H-DeviceContextChange") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      SUCCEED("deviceCount less then 2");
    } else {
      int canAccessPeer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
      if (canAccessPeer) {
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipMemcpy(A_d, A_h, NUM_ELM*sizeof(TestType),
                            hipMemcpyDefault));
        HIP_CHECK(hipMemcpy(B_d, A_d, NUM_ELM*sizeof(TestType),
                            hipMemcpyDefault));
        HIP_CHECK(hipMemcpy(B_h, B_d, NUM_ELM*sizeof(TestType),
                            hipMemcpyDefault));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
      } else {
        SUCCEED("P2P capability is not present");
      }
    }
  }

  SECTION("H2D-D2D-D2H-PeerGPU") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      SUCCEED("deviceCount less then 2");
    } else {
      int canAccessPeer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
      if (canAccessPeer) {
        HIP_CHECK(hipSetDevice(1));
        TestType *C_d{nullptr};
        HipTest::initArrays<TestType>(nullptr, nullptr, &C_d,
            nullptr, nullptr, nullptr,
            NUM_ELM*sizeof(TestType));
        HIP_CHECK(hipMemcpy(A_d, A_h, NUM_ELM*sizeof(TestType),
                            hipMemcpyDefault));
        HIP_CHECK(hipMemcpy(C_d, A_d, NUM_ELM*sizeof(TestType),
                            hipMemcpyDefault));
        HIP_CHECK(hipMemcpy(B_h, C_d, NUM_ELM*sizeof(TestType),
                            hipMemcpyDefault));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        HIP_CHECK(hipFree(C_d));
      } else {
        SUCCEED("P2P capability is not present");
      }
    }
  }

  HipTest::freeArrays<TestType>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_Ph,
                                B_Ph, nullptr, true);
}
/*
This testcase verifies the multi thread scenario
*/
TEST_CASE("Unit_hipMemcpy_MultiThreadWithSerialization") {
  HIP_CHECK(hipDeviceReset());

  // Simplest cases: serialize the threads, and also used pinned memory:
  // This verifies that the sub-calls to memcpytest2 are correct.
  multiThread_1<float>(true, true);

  // Serialize, but use unpinned memory to stress the unpinned memory xfer path.
  multiThread_1<float>(true, false);
}

/*
This testcase verifies hipMemcpy API with pinnedMemory and hostRegister
along with kernel launches
*/

TEMPLATE_TEST_CASE("Unit_hipMemcpy_PinnedRegMemWithKernelLaunch",
                   "", int, float, double) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    SUCCEED("No of devices are less than 2");
  } else {
    // 1 refers to pinned Memory
    // 2 refers to register Memory
    int MallocPinType = GENERATE(0, 1);
    size_t Nbytes = NUM_ELM * sizeof(TestType);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                                            threadsPerBlock, NUM_ELM);

    TestType *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
    TestType *X_d{nullptr}, *Y_d{nullptr}, *Z_d{nullptr};
    TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
    if (MallocPinType) {
      HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM, true);
    } else {
      A_h = reinterpret_cast<TestType*>(malloc(Nbytes));
      HIP_CHECK(hipHostRegister(A_h, Nbytes, hipHostRegisterDefault));
      B_h = reinterpret_cast<TestType*>(malloc(Nbytes));
      HIP_CHECK(hipHostRegister(B_h, Nbytes, hipHostRegisterDefault));
      C_h = reinterpret_cast<TestType*>(malloc(Nbytes));
      HIP_CHECK(hipHostRegister(C_h, Nbytes, hipHostRegisterDefault));
      HipTest::initArrays<TestType>(&A_d, &B_d, &C_d, nullptr, nullptr,
                                    nullptr, NUM_ELM, false);
      HipTest::setDefaultData<TestType>(NUM_ELM, A_h, B_h, C_h);
    }
    HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, 0, static_cast<const TestType*>(A_d),
                       static_cast<const TestType*>(B_d), C_d, NUM_ELM);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h, B_h, C_h, NUM_ELM);

    unsigned int seed = time(0);
    HIP_CHECK(hipSetDevice(HipTest::RAND_R(&seed) % (numDevices-1)+1));

    int device;
    HIP_CHECK(hipGetDevice(&device));
    std::cout <<"hipMemcpy is set to happen between device 0 and device "
      <<device << std::endl;
    HipTest::initArrays<TestType>(&X_d, &Y_d, &Z_d, nullptr,
                                  nullptr, nullptr, NUM_ELM, false);

    for (int j = 0; j < NUM_ELM; j++) {
      A_h[j] = 0;
      B_h[j] = 0;
      C_h[j] = 0;
    }

    HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(X_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_h, B_d, Nbytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(Y_d, B_h, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, 0, static_cast<const TestType*>(X_d),
                       static_cast<const TestType*>(Y_d), Z_d, NUM_ELM);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(C_h, Z_d, Nbytes, hipMemcpyDeviceToHost));

    HipTest::checkVectorADD(A_h, B_h, C_h, NUM_ELM);

    if (MallocPinType) {
      HipTest::freeArrays<TestType>(A_d, B_d, C_d, A_h, B_h, C_h, true);
    } else {
      HIP_CHECK(hipHostUnregister(A_h));
      free(A_h);
      HIP_CHECK(hipHostUnregister(B_h));
      free(B_h);
      HIP_CHECK(hipHostUnregister(C_h));
      free(C_h);
      HipTest::freeArrays<TestType>(A_d, B_d, C_d, nullptr,
                                    nullptr, nullptr, false);
    }
      HipTest::freeArrays<TestType>(X_d, Y_d, Z_d, nullptr,
                                    nullptr, nullptr, false);
  }
}
