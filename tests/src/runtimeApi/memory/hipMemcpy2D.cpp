/*
   Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */

// Testcase Description: This test case achieves three scenarios
// 1) Verifies the working of Memcpy2D API negative scenarios by
//    Pass NULL to destination pointer
//    Pass NULL to Source pointer
//    Pass NULL to both Source and destination pointers
//    Pass same pointer to both source and destination pointers.
//    Pass width greater than spitch/dpitch
// 2) Verifies hipMemcpy2D API by
//    pass 0 to destionation pitch
//    pass 0 to source pitch
//    pass 0 to both source and destination pitches
//    pass 0 to width
//    pass 0 to height
// 3) Verifies working of Memcpy2D API by performing D2H and
//    H2D memory kind copies
// 4) Verifies working of Memcpy2D API by performing D2D
//    in same GPU device and the peer GPU device.
// 5) Verify hipMemcpy2D API on pinned host memory in same and peer GPU devices


/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST_NAMED: %t hipMemcpy2D_NegativeTest  --tests 1
 * TEST_NAMED: %t hipMemcpy2D_H2D_D2H  --tests 3
 * TEST_NAMED: %t hipMemcpy2D_D2D  --tests 4
 * TEST_NAMED: %t hipMemcpy2D_PinnedMemory --tests 5
 * HIT_END
 */

#include "test_common.h"

#define NUM_H 256
#define NUM_W 256
#define COLUMNS 8
#define ROWS 8


class Memcpy2D {
  char *A_h{nullptr}, *C_h{nullptr}, *A_d{nullptr},
       *B_h{nullptr}, *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(char)};
  size_t sizeElements{width * NUM_H};
  size_t elements{NUM_W * NUM_H};
  bool ValidateResult(char *result, int compare);
 public:
  void AllocateMemory();
  void DeAllocateMemory();
  bool Memcpy2D_NegativeTest();
  bool Memcpy2D_NegativeTest_SizeCheck();
  bool Memcpy2D_H2D_D2HKind();
  bool Memcpy2D_D2DKind_SameGPU();
  bool Memcpy2D_D2DKind_MultiGPU();
  bool Memcpy2D_PinnedMemory_SameGPU();
  bool Memcpy2D_PinnedMemory_MultiGPU();
};

bool Memcpy2D::ValidateResult(char *result, int compare) {
  int count = 0;
  for (int row = 0; row < ROWS; row++) {
    for (int column = 0; column < COLUMNS; column++) {
      if (result[(row * NUM_H) + column] != compare) {
         return false;
      }
      ++count;
    }
  }
  return true;
}

void Memcpy2D::AllocateMemory() {
  A_h = reinterpret_cast<char *>(malloc(sizeElements));
  HIPASSERT(A_h != nullptr);
  B_h = reinterpret_cast<char *>(malloc(sizeElements));
  HIPASSERT(B_h != nullptr);
  C_h = reinterpret_cast<char *>(malloc(sizeElements));
  HIPASSERT(C_h != nullptr);
  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
        &pitch_A, width, NUM_H));
  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d),
        &pitch_B, width, NUM_H));
  for (size_t i=0; i < elements; i++) {
    A_h[i] = 3;
    B_h[i] = 4;
    C_h[i] = 123;
  }
}

void Memcpy2D::DeAllocateMemory() {
  HIPCHECK(hipFree(A_d)); HIPCHECK(hipFree(B_d));
  free(A_h); free(B_h); free(C_h);
}


bool Memcpy2D::Memcpy2D_H2D_D2HKind() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool testResult = true;
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  // hipMemcpy Device to Host
  HIPCHECK(hipMemcpy2D(A_h, width, A_d, pitch_A,
        COLUMNS, ROWS, hipMemcpyDeviceToHost));
  testResult = ValidateResult(A_h, memsetval);
  // hipMemcpy Host to Device and validating
  // the result by copying the device data to host data
  HIPCHECK(hipMemcpy2D(B_d, pitch_B, B_h, width,
        COLUMNS, ROWS, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy2D(C_h, width, B_d, pitch_B,
        COLUMNS, ROWS, hipMemcpyDeviceToHost));
  testResult &= ValidateResult(C_h, B_h[0]);
  DeAllocateMemory();
  return testResult;
}

bool Memcpy2D::Memcpy2D_PinnedMemory_SameGPU() {
  HIPCHECK(hipSetDevice(0));
  bool testResult = true;
  AllocateMemory();
  char *D_h{nullptr};
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), sizeElements));
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  HIPCHECK(hipMemcpy2D(D_h, width, A_d, pitch_A,
        COLUMNS, ROWS, hipMemcpyDeviceToHost));
  testResult = ValidateResult(D_h, memsetval);
  DeAllocateMemory();
  HIPCHECK(hipHostFree(D_h));
  return testResult;
}

bool Memcpy2D::Memcpy2D_PinnedMemory_MultiGPU() {
  bool testResult = true;
  int numDevices = 0;
  int canAccessPeer = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    hipDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    // Check for peer devices and performing D2D on the devices
    if (canAccessPeer) {
      HIPCHECK(hipSetDevice(0));
      AllocateMemory();
      HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
      HIPCHECK(hipSetDevice(1));
      char *D_h{nullptr};
      HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), sizeElements));
      HIPCHECK(hipMemcpy2D(D_h, width, A_d, pitch_A,
            COLUMNS, ROWS, hipMemcpyDeviceToHost));
      testResult = ValidateResult(D_h, memsetval);
      DeAllocateMemory();
      HIPCHECK(hipHostFree(D_h));
    } else {
      printf("Machine does not seem to have P2P Capabilities, Empty Pass");
    }
  } else {
    printf("Testcase Skipped as no of devices < 2");
  }
  return testResult;
}

bool Memcpy2D::Memcpy2D_D2DKind_SameGPU() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool testResult = true;
  // Performs D2D on same GPU device
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  HIPCHECK(hipMemcpy2D(B_d, pitch_B, A_d,
        pitch_A, COLUMNS, ROWS, hipMemcpyDeviceToDevice));
  HIPCHECK(hipMemcpy2D(B_h, width, B_d, pitch_B,
        COLUMNS, ROWS, hipMemcpyDeviceToHost));
  testResult = ValidateResult(B_h, memsetval);
  DeAllocateMemory();
  return testResult;
}
bool Memcpy2D::Memcpy2D_D2DKind_MultiGPU() {
  int numDevices = 0;
  bool testResult = true;
  int canAccessPeer = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    for (int j =1; j < numDevices; j++) {
      hipDeviceCanAccessPeer(&canAccessPeer, 0, j);
      // Check for peer devices and performing D2D on the devices
      if (canAccessPeer) {
        HIPCHECK(hipSetDevice(0));
        AllocateMemory();
        HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
        HIPCHECK(hipSetDevice(j));
        char *X_d{nullptr};
        size_t pitch_X;
        HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&X_d),
              &pitch_X, width, NUM_H));
        HIPCHECK(hipMemcpy2D(X_d, pitch_X, A_d,
              pitch_A, COLUMNS, ROWS, hipMemcpyDeviceToDevice));
        HIPCHECK(hipMemcpy2D(C_h, width, X_d,
              pitch_X, COLUMNS, ROWS, hipMemcpyDeviceToHost));
        testResult &= ValidateResult(C_h, memsetval);
        HIPCHECK(hipFree(X_d));
        DeAllocateMemory();
      } else {
        printf("Machine does not seem to have P2P between 0 & %d", j);
      }
    }
  } else {
    printf("skipped the testcase as no of devices is less than 2");
  }
  return testResult;
}

bool Memcpy2D::Memcpy2D_NegativeTest() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool TestPassed = true;
  hipError_t err;
  err = hipMemcpy2D(A_h, width, nullptr,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2D failed when source pointer are null");
    TestPassed = false;
  }
  // hipMemcpy2D API by Passing nullptr to destination
  err = hipMemcpy2D(nullptr, width, A_d,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2D failed when  dest pointer are null");
    TestPassed = false;
  }
  // hipMemcpy2D by Passing nullptr to both Source and Destination ptr
  err = hipMemcpy2D(nullptr, width, nullptr,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2D failed when both source and dest pointer are null");
    TestPassed = false;
  }
  // hipMemcpy2D API where width is greater than destination pitch
  err = hipMemcpy2D(A_h, 10, A_d, pitch_A,
      NUM_W, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2D failed where width is greater than destination pitch");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}

bool Memcpy2D::Memcpy2D_NegativeTest_SizeCheck() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool TestPassed = true;
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  hipError_t err;
  // hipMemcpy2D API where Destination Pitch is zero
  err = hipMemcpy2D(A_h, 0, A_d,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2D failed when source pitch is null");
    TestPassed = false;
  }
  // hipMemcpy2D API where Source Pitch is zero
  err = hipMemcpy2D(A_h, width, A_d,
      0, NUM_W, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2D failed when destination pitch is null");
    TestPassed = false;
  }
  // hipMemcpy2D API where Source and Destination Pitch are zero
  err = hipMemcpy2D(A_h, 0, A_d,
      0, NUM_W, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    printf("hipMemcpy2D failed when source and destination pitches are null");
    TestPassed = false;
  }
  // hipMemcpy2D API where height is zero
  // hipMemcpy2D API would return success for width and height as 0
  // Validating the result with the initialized value
  err = hipMemcpy2D(A_h, width, A_d,
      pitch_A, NUM_W, 0, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    TestPassed = ValidateResult(A_h, 3);
  } else {
    printf("hipMemcpy2D failed when Height is null");
    TestPassed = false;
  }
  // hipMemcpy2D API where width is zero
  // hipMemcpy2D API would return success for width and height as 0
  // Validating the result with the initialized value
  err = hipMemcpy2D(A_h, width, A_d,
      pitch_A, 0, NUM_H, hipMemcpyDeviceToHost);
  if (err == hipSuccess) {
    TestPassed = ValidateResult(A_h, 3);
  } else {
    printf("hipMemcpy2D failed when Width is null");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}

int main(int argc, char* argv[]) {
  bool TestPassed = true;
  Memcpy2D Memcpy2DObj;
  HipTest::parseStandardArguments(argc, argv, false);
  if (p_tests == 1) {
    TestPassed &= Memcpy2DObj.Memcpy2D_NegativeTest();
  } else if (p_tests == 2) {
    TestPassed &= Memcpy2DObj.Memcpy2D_NegativeTest_SizeCheck();
  } else if (p_tests == 3) {
    TestPassed &= Memcpy2DObj.Memcpy2D_H2D_D2HKind();
  } else if (p_tests == 4) {
    TestPassed &= Memcpy2DObj.Memcpy2D_D2DKind_SameGPU();
    TestPassed &= Memcpy2DObj.Memcpy2D_D2DKind_MultiGPU();
  } else if (p_tests == 5) {
    TestPassed &= Memcpy2DObj.Memcpy2D_PinnedMemory_SameGPU();
    TestPassed &= Memcpy2DObj.Memcpy2D_PinnedMemory_MultiGPU();
  } else {
    failed("Didnt receive any valid option. Try options 1 to 5\n");
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Failed!");
  }
}
