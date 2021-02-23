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
 * Scenario 1 :
 * 1. hipMemcpy2DFromArray simple scenarios
 * 2. Extent Validation Scenarios
 * 3. Device context Change
 * 4. Negative Scenarios
 * 5. Pinned Host Memory from same and Peer GPU.
 */
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t hipMemcpy2DFromArray_simple --tests 1
 * TEST_NAMED: %t hipMemcpy2DFromArray_ExtentValidation --tests 2
 * TEST_NAMED: %t hipMemcpy2DFromArray_DeviceContextChange --tests 3
 * TEST_NAMED: %t hipMemcpy2DFromArray_NegativeTests --tests 4
 * TEST_NAMED: %t hipMemcpy2DFromArray_PinnedHostMemory --tests 5
 * HIT_END
 */
#include "test_common.h"

#define NUM_W 10
#define NUM_H 10
#define INITIAL_VAL 8

template<typename T>
class Memcpy2DFromArray {
  hipArray *A_d{nullptr};
  T *hData{nullptr}, *A_h{nullptr};
  size_t width, height;
  size_t elements{NUM_W * NUM_H};
  hipError_t err;
 public:
  void AllocateMemory();
  void DeAllocateMemory();
  bool hipMemcpy2DFromArray_NegativeTests();
  bool hipMemcpy2DFromArray_simple();
  bool hipMemcpy2DFromArray_SizeCheck();
  bool hipMemcpy2DFromArray_PeerDeviceContext();
  bool hipMemcpy2DFromArray_PinnedHostMemory_SameGPU();
  bool hipMemcpy2DFromArray_PinnedHostMemory_PeerGPU();
  bool ValidateResult(T* result, T compare);
};
template <typename T>
void Memcpy2DFromArray<T>::AllocateMemory() {
  width = NUM_W * sizeof(T);
  height = NUM_H;
  hData = reinterpret_cast<T*>(malloc(width * NUM_H));
  A_h = reinterpret_cast<T*>(malloc(width * NUM_H));
  for (int i = 0; i < elements; i++) {
    A_h[i] = 1;
    hData[i] = INITIAL_VAL;
  }
  hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
  HIPCHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
  HIPCHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, width,
                              width, NUM_H, hipMemcpyHostToDevice));
}
template <typename T>
bool Memcpy2DFromArray<T>::ValidateResult(T *result, T compare) {
  bool TestPassed = true;
  for (int i = 0; i < NUM_W; i++) {
    for (int j = 0; j < NUM_H; j++) {
      if (result[(i*NUM_H) + j] != compare) {
        TestPassed = false;
      }
    }
  }
  return TestPassed;
}
template <typename T>
void Memcpy2DFromArray<T>::DeAllocateMemory() {
  hipFreeArray(A_d);
  free(hData);
  free(A_h);
}

template <typename T>
bool Memcpy2DFromArray<T>::hipMemcpy2DFromArray_PinnedHostMemory_SameGPU() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  T *D_h{nullptr};
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), width * NUM_H));
  err = hipMemcpy2DFromArray(D_h, width, A_d,
                             0, 0, width,
                             NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    TestPassed = ValidateResult(D_h, INITIAL_VAL);
  } else {
    printf("hipMemcpy2DFromArray failed for PinnedHostMemory same GPU\n");
    TestPassed = false;
  }
  DeAllocateMemory();
  HIPCHECK(hipHostFree(D_h));
  return TestPassed;
}

template <typename T>
bool Memcpy2DFromArray<T>::hipMemcpy2DFromArray_PinnedHostMemory_PeerGPU() {
  bool TestPassed = true;
  int canAccessPeer = 0;
  HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
  // Check for peer devices and performing D2D on the devices
  if (canAccessPeer) {
    HIPCHECK(hipSetDevice(0));
    AllocateMemory();
    HIPCHECK(hipSetDevice(1));
    T *D_h{nullptr};
    HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), width * NUM_H));
    err = hipMemcpy2DFromArray(D_h, width, A_d,
                               0, 0, width,
                               NUM_H, hipMemcpyDeviceToHost);
    if (err == hipSuccess) {
      TestPassed = ValidateResult(D_h, INITIAL_VAL);
    } else {
      printf("hipMemcpy2DFromArray failed for PinnedHostMemory Peer GPU\n");
      TestPassed = false;
    }
    DeAllocateMemory();
    HIPCHECK(hipHostFree(D_h));
  } else {
    printf("Machine does not seem to have P2P Capabilities, Empty Pass");
  }
  return TestPassed;
}

template <typename T>
bool Memcpy2DFromArray<T>::hipMemcpy2DFromArray_simple() {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  err = hipMemcpy2DFromArray(A_h, width, A_d,
                             0, 0, width, NUM_H,
                             hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    TestPassed = ValidateResult(A_h, INITIAL_VAL);
  } else {
    printf("hipMemcpy2DFromArray failed for simple copy\n");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}
template <typename T>
bool Memcpy2DFromArray<T>::hipMemcpy2DFromArray_PeerDeviceContext() {
  bool TestPassed = true;
  int peerAccess = 0;
  HIPCHECK(hipDeviceCanAccessPeer(&peerAccess, 0, 1));
  if (!peerAccess) {
    printf("Skipped the test as there is no peer access\n");
  } else {
    HIPCHECK(hipSetDevice(0));
    AllocateMemory();
    HIPCHECK(hipSetDevice(1));
    err = hipMemcpy2DFromArray(A_h, width, A_d,
                               0, 0, width,
                               NUM_H, hipMemcpyDeviceToHost);
    if (err == hipSuccess) {
      TestPassed = ValidateResult(A_h, INITIAL_VAL);
    } else {
      printf("hipMemcpy2DFromArray failed for peer device context\n");
      TestPassed = false;
    }
    DeAllocateMemory();
  }
  return TestPassed;
}

template <typename T>
bool Memcpy2DFromArray<T>::hipMemcpy2DFromArray_SizeCheck() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool TestPassed = true;
  // hipMemcpy2DFromArray API where Destination width is 0
  err = hipMemcpy2DFromArray(A_h, 0, A_d,
                             0, 0, NUM_W*sizeof(T),
                             NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2DFromArray failed when destination width is zero");
    TestPassed = false;
  }

  // hipMemcpy2DFromArray API where height is zero
  // hipMemcpy2DFromArray API would return success for width and height as 0
  // Validating the result with the initialized value
  err = hipMemcpy2DFromArray(A_h, width, A_d,
                             0, 0, NUM_W*sizeof(T),
                             0, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    TestPassed &= ValidateResult(A_h, 1);
  } else {
    printf("hipMemcpy2DFromArray failed when Height is null");
    TestPassed = false;
  }
  // hipMemcpy2DFromArray API where width is zero
  // hipMemcpy2DFromArray API would return success for width and height as 0
  // Validating the result with the initialized value
  err = hipMemcpy2DFromArray(A_h, width, A_d,
                             0, 0, 0, NUM_H,
                             hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    TestPassed &= ValidateResult(A_h, 1);
  } else {
    printf("hipMemcpy2DFromArray failed when Width is null");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}

template <typename T>
bool Memcpy2DFromArray<T>::hipMemcpy2DFromArray_NegativeTests() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool TestPassed = true;
  // Passing nullptr to destination
  err = hipMemcpy2DFromArray(nullptr, width, A_d,
                             0, 0, width, NUM_H,
                             hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2DFromArray failed when  dest pointer are null");
    TestPassed = false;
  }
  // Passing nullptr to source
  err = hipMemcpy2DFromArray(A_h, width, nullptr,
                             0, 0, width, NUM_H,
                             hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2DFromArray failed when source pointer are null");
    TestPassed = false;
  }
  // Passing offset 1 and trying to perform array out of bounds
  err = hipMemcpy2DFromArray(A_h, width, A_d, 1,
                             1, width, NUM_H,
                             hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2DFromArray failed offset 1 and perform full copy");
    TestPassed = false;
  }
  //  Copying array more than allocated (array out of bounds)
  err = hipMemcpy2DFromArray(A_h, width, A_d, 0,
                             0, width+2, NUM_H+2,
                             hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2DFromArray failed where array is out of bound");
    TestPassed = false;
  }

  DeAllocateMemory();
  return TestPassed;
}


int main(int argc, char **argv) {
  bool TestPassed = true;
  HipTest::parseStandardArguments(argc, argv, false);
  Memcpy2DFromArray<float> Array_obj;
  int numDevices = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  if (p_tests == 1) {
    TestPassed = Array_obj.hipMemcpy2DFromArray_simple();
  } else if (p_tests == 2) {
    TestPassed &= Array_obj.hipMemcpy2DFromArray_SizeCheck();
  } else if (p_tests == 3) {
    if (numDevices > 1) {
    TestPassed &= Array_obj.hipMemcpy2DFromArray_PeerDeviceContext();
    } else {
      printf("skipped the testcase as noof devices <2\n");
    }
  } else if (p_tests == 4) {
    TestPassed &= Array_obj.hipMemcpy2DFromArray_NegativeTests();
  } else if (p_tests == 5) {
    if (numDevices > 1) {
    TestPassed &= Array_obj.hipMemcpy2DFromArray_PinnedHostMemory_SameGPU();
    TestPassed &= Array_obj.hipMemcpy2DFromArray_PinnedHostMemory_PeerGPU();
    } else {
      printf("skipped the testcases as noof devices <2\n");
    }
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
