/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
 * Test Scenarios:
 * 1. Perform simple hipMemcpyAtoH
 * 2. Perform bytecount 0  validation for hipMemcpyAtoH API
 * 3. Allocate Memory from one GPU device and call hipMemcpyAtoH from Peer
 *    GPU device
 * 4. Perform hipMemcpyAtoH Negative Scenarios
 * 5. Perform hipMemcpyAtoH on Pinned Host memory
 * Scenarios 2 is disabled as there is a corresponding bug raised for it.
 */
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t hipMemcpyAtoH_simple --tests 1
 * TEST_NAMED: %t hipMemcpyAtoH_DeviceContextChange --tests 3 EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipMemcpyAtoH_NegativeTests --tests 4
 * TEST_NAMED: %t hipMemcpyAtoH_PinnedHostMemory --tests 5
 * HIT_END
 */
#include "test_common.h"

#define NUM_W 10
#define NUM_H 1
#define INITIAL_VAL 8
#define BYTE_COUNT 2
template<typename T>
class MemcpyAtoH {
  hipArray *A_d;
  T *hData, *B_h;
  size_t width;
  size_t height;
 public:
  void AllocateMemory();
  void DeAllocateMemory();
  bool hipMemcpyAtoH_NegativeTests();
  bool hipMemcpyAtoH_simple();
  bool hipMemcpyAtoH_PinnedHostMemory();
  bool hipMemcpyAtoH_ByteCountZero();
  bool hipMemcpyAtoH_PeerDeviceContext();
  bool ValidateResult(T* result, T compare);
};
template <typename T>
void MemcpyAtoH<T>::AllocateMemory() {
  width = NUM_W * sizeof(T);
  height = NUM_H;
  hData = reinterpret_cast<T*>(malloc(width));
  B_h = reinterpret_cast<T*>(malloc(width));
  for (int i = 0; i < NUM_W; i++) {
    B_h[i] = 10;
    hData[i] = INITIAL_VAL;
  }
  hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
  HIPCHECK(hipMallocArray(&A_d, &desc, NUM_W, 1, hipArrayDefault));
  HIPCHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, sizeof(T)*NUM_W,
                              sizeof(T)*NUM_W, 1, hipMemcpyHostToDevice));
}
template <typename T>
bool MemcpyAtoH<T>::ValidateResult(T *result, T compare) {
  bool TestPassed = true;
  for (int i = 0; i < BYTE_COUNT; i++) {
    if (result[i] != compare) {
      TestPassed = false;
      break;
    }
  }
  return TestPassed;
}
template <typename T>
void MemcpyAtoH<T>::DeAllocateMemory() {
  hipFreeArray(A_d);
  free(hData);
  free(B_h);
}
template <typename T>
bool MemcpyAtoH<T>::hipMemcpyAtoH_simple() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  HIPCHECK(hipMemcpyAtoH(B_h, A_d, 0, BYTE_COUNT*sizeof(T)));
  TestPassed = ValidateResult(B_h, hData[0]);
  DeAllocateMemory();
  return TestPassed;
}

template <typename T>
bool MemcpyAtoH<T>::hipMemcpyAtoH_PinnedHostMemory() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  T *D_h{nullptr};
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), width * NUM_H));
  HIPCHECK(hipMemcpyAtoH(D_h, A_d, 0, BYTE_COUNT*sizeof(T)));
  TestPassed = ValidateResult(D_h, hData[0]);
  HIPCHECK(hipHostFree(D_h));
  DeAllocateMemory();
  return TestPassed;
}

template <typename T>
bool MemcpyAtoH<T>::hipMemcpyAtoH_PeerDeviceContext() {
  bool TestPassed = true;
  int peerAccess = 0;
  int numDevices = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIPCHECK(hipDeviceCanAccessPeer(&peerAccess, 0, 1));
    if (!peerAccess) {
      printf("Skipped the test as there is no peer access\n");
    } else {
      HIPCHECK(hipSetDevice(0));
      AllocateMemory();
      HIPCHECK(hipSetDevice(1));
      HIPCHECK(hipMemcpyAtoH(B_h, A_d, 0, BYTE_COUNT*sizeof(T)));
      TestPassed = ValidateResult(B_h, hData[0]);
      DeAllocateMemory();
    }
  } else {
    printf("Testcase Skipped as no of devices < 2");
  }

  return TestPassed;
}
template <typename T>
bool MemcpyAtoH<T>::hipMemcpyAtoH_ByteCountZero() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  hipError_t err;
  err = hipMemcpyAtoH(B_h, A_d, 0, 0);
  if (err == hipSuccess) {
    TestPassed = ValidateResult(B_h, 10);
  } else {
    printf("hipMemcpyAtoH failed when byteCount is 0 \n");
    TestPassed = false;
  }
  // Source Array is nullptr
  err = hipMemcpyAtoH(B_h, nullptr, 0, BYTE_COUNT*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyAtoH failed when src array is nullptr\n");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}
template <typename T>
bool MemcpyAtoH<T>::hipMemcpyAtoH_NegativeTests() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  hipError_t err;
  // Destination pointer is nullptr
  err = hipMemcpyAtoH(nullptr, A_d, 0, BYTE_COUNT*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyAtoH failed when dest ptr is nullptr\n");
    TestPassed = false;
  }
  // Source offset is more than allocated size
  err = hipMemcpyAtoH(B_h, A_d, 100, BYTE_COUNT*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyAtoH failed when source offset invalid\n");
    TestPassed = false;
  }
  // ByteCount is greater than allocated size
  err = hipMemcpyAtoH(B_h, A_d, 0, 12*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyAtoH failed when byteCount > allocatedSize\n");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}


int main(int argc, char **argv) {
  bool TestPassed = true;
  HipTest::parseStandardArguments(argc, argv, false);
  MemcpyAtoH<float> AtoH_obj;
  if (p_tests == 1) {
    TestPassed = AtoH_obj.hipMemcpyAtoH_simple();
  } else if (p_tests == 2) {
    TestPassed = AtoH_obj.hipMemcpyAtoH_ByteCountZero();
  } else if (p_tests == 3) {
    TestPassed = AtoH_obj.hipMemcpyAtoH_PeerDeviceContext();
  } else if (p_tests == 4) {
    TestPassed = AtoH_obj.hipMemcpyAtoH_NegativeTests();
  } else if (p_tests == 5) {
    TestPassed = AtoH_obj.hipMemcpyAtoH_PinnedHostMemory();
  } else {
    printf("Provide a valid option \n");
    TestPassed = false;
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Failed!");
  }
}
