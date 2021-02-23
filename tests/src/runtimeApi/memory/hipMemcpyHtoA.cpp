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
 * Test Scenarios:
 * 1. Perform simple hipMemcpyHtoA
 * 2. Perform bytecount 0  validation for hipMemcpyHtoA API
 * 3. Allocate Memory from one GPU device and call hipMemcpyHtoA from Peer
 *    GPU device
 * 4. Perform hipMemcpyHtoA Negative Scenarios
 * 5. Perform hipMemcpyHtoA on Pinned Host memory
 * Scenarios 2 is disabled as there is a corresponding bug raised for it.
*/
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t hipMemcpyHtoA_simple --tests 1
 * TEST_NAMED: %t hipMemcpyHtoA_DeviceContextChange --tests 3 EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipMemcpyHtoA_NegativeTests --tests 4
 * TEST_NAMED: %t hipMemcpyHtoA_PinnedHostMemory --tests 5
 * HIT_END
 */
#include "test_common.h"

#define NUM_W 10
#define NUM_H 1
#define INITIAL_VAL 8
#define BYTECOUNT 2

template<typename T>
class MemcpyHtoA {
  hipArray *A_d;
  T *hData, *B_h, *A_h, *D_h;
  size_t width;
  size_t height;
 public:
  void AllocateMemory();
  void DeAllocateMemory();
  bool hipMemcpyHtoA_NegativeTests();
  bool hipMemcpyHtoA_simple();
  bool hipMemcpyHtoA_PinnedHostMemory();
  bool hipMemcpyHtoA_ByteCountZero();
  bool hipMemcpyHtoA_PeerDeviceContext();
  bool ValidateResult(T* result, T compare);
};
template <typename T>
void MemcpyHtoA<T>::AllocateMemory() {
  width = NUM_W * sizeof(T);
  height = NUM_H;
  hData = reinterpret_cast<T*>(malloc(width));
  B_h = reinterpret_cast<T*>(malloc(width));
  A_h = reinterpret_cast<T*>(malloc(width));
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), width * NUM_H));
  for (int i = 0; i < NUM_W; i++) {
    A_h[i] = 1;
    B_h[i] = 10;
    D_h[i] = 123;
    hData[i] = INITIAL_VAL;
  }
  hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
  HIPCHECK(hipMallocArray(&A_d, &desc, NUM_W, 1, hipArrayDefault));
  HIPCHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, sizeof(T)*NUM_W,
                              sizeof(T)*NUM_W, 1, hipMemcpyHostToDevice));
}
template <typename T>
bool MemcpyHtoA<T>::ValidateResult(T *result, T compare) {
  bool TestPassed = true;
  for (int i = 0; i < BYTECOUNT; i++) {
    if (result[i] != compare) {
      TestPassed = false;
      break;
    }
  }
  return TestPassed;
}
template <typename T>
void MemcpyHtoA<T>::DeAllocateMemory() {
  hipFreeArray(A_d);
  free(hData);
  free(B_h);
  free(A_h);
}
template <typename T>
bool MemcpyHtoA<T>::hipMemcpyHtoA_simple() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  HIPCHECK(hipMemcpyHtoA(A_d, 0, B_h, BYTECOUNT*sizeof(T)));
  HIPCHECK(hipMemcpy2DFromArray(A_h, sizeof(T)*NUM_W, A_d,
           0, 0, sizeof(T)*NUM_W, 1, hipMemcpyDeviceToHost));
  TestPassed = ValidateResult(A_h, B_h[0]);
  DeAllocateMemory();
  return TestPassed;
}
template <typename T>
bool MemcpyHtoA<T>::hipMemcpyHtoA_PinnedHostMemory() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  HIPCHECK(hipMemcpyHtoA(A_d, 0, D_h, BYTECOUNT*sizeof(T)));
  HIPCHECK(hipMemcpy2DFromArray(A_h, sizeof(T)*NUM_W, A_d,
           0, 0, sizeof(T)*NUM_W, 1, hipMemcpyDeviceToHost));
  TestPassed = ValidateResult(A_h, D_h[0]);
  DeAllocateMemory();
  HIPCHECK(hipHostFree(D_h));
  return TestPassed;
}

template <typename T>
bool MemcpyHtoA<T>::hipMemcpyHtoA_PeerDeviceContext() {
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
      HIPCHECK(hipMemcpyHtoA(A_d, 0, B_h, BYTECOUNT*sizeof(T)));
      HIPCHECK(hipMemcpy2DFromArray(A_h, sizeof(T)*NUM_W, A_d,
      0, 0, sizeof(T)*NUM_W, 1, hipMemcpyDeviceToHost));
      TestPassed = ValidateResult(A_h, B_h[0]);
      DeAllocateMemory();
    }
  } else {
    printf("Testcase Skipped as no of devices < 2");
  }
  return TestPassed;
}
template <typename T>
bool MemcpyHtoA<T>::hipMemcpyHtoA_ByteCountZero() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  hipError_t err;
  err = hipMemcpyHtoA(A_d, 0, B_h, 0);
  HIPCHECK(hipMemcpy2DFromArray(A_h, sizeof(T)*NUM_W, A_d,
           0, 0, sizeof(T)*NUM_W, 1, hipMemcpyDeviceToHost));
  if (err == hipSuccess) {
    TestPassed = ValidateResult(A_h, INITIAL_VAL);
  } else {
    printf("hipMemcpyHtoA failed when byteCount is 0 \n");
    TestPassed = false;
  }
  // Destination Array is nullptr
  err = hipMemcpyHtoA(nullptr, 0, B_h, BYTECOUNT*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyHtoA failed when dest ptr is nullptr\n");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}

template <typename T>
bool MemcpyHtoA<T>::hipMemcpyHtoA_NegativeTests() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  hipError_t err;
  // Source pinter is nullptr
  err = hipMemcpyHtoA(A_d, 0, nullptr, BYTECOUNT*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyHtoA failed when src array is nullptr\n");
    TestPassed = false;
  }
  // dst offset is more than allocated size
  err = hipMemcpyHtoA(A_d, 100, B_h, BYTECOUNT*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyHtoA failed when source offset invalid\n");
    TestPassed = false;
  }
  // ByteCount is greater than allocated size
  err = hipMemcpyHtoA(A_d, 0, B_h, 12*sizeof(T));
  if (err == hipSuccess) {
    printf("hipMemcpyHtoA failed when byteCount > allocatedSize\n");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}


int main(int argc, char **argv) {
  bool TestPassed = true;
  HipTest::parseStandardArguments(argc, argv, false);
  MemcpyHtoA<float> HtoA_obj;
  if (p_tests == 1) {
    TestPassed = HtoA_obj.hipMemcpyHtoA_simple();
  } else if (p_tests == 2) {
    TestPassed = HtoA_obj.hipMemcpyHtoA_ByteCountZero();
  } else if (p_tests == 3) {
    TestPassed = HtoA_obj.hipMemcpyHtoA_PeerDeviceContext();
  } else if (p_tests == 4) {
    TestPassed = HtoA_obj.hipMemcpyHtoA_NegativeTests();
  } else if (p_tests == 5) {
    TestPassed = HtoA_obj.hipMemcpyHtoA_PinnedHostMemory();
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
