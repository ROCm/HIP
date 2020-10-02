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
// 1) Verifies the working of Memcpy2DAsync API negative scenarios by
//    Pass NULL to destination pointer
//    Pass NULL to Source pointer
//    Pass NULL to both Source and destination pointers
//    Pass same pointer to both source and destination pointers.
//    Pass width greater than spitch/dpitch
// 2) Verifies hipMemcpy2DAsync API by
//    pass 0 to destionation pitch
//    pass 0 to source pitch
//    pass 0 to both source and destination pitches
//    pass 0 to width
//    pass 0 to height
// 3) Verifies working of Memcpy2DAsync API by performing D2H
//    and H2D memory kind copies
// 4) Verifies working of Memcpy2DAsync API by performing D2D
//    on same GPU device and the peer GPU device.
// 5) Verifies working hipMemcpy2DAsync API along with launching Kernel
// 6) Veirfy hipMemcpy2DAsync by allocating pinned host memory

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * TEST_NAMED: %t hipMemcpy2DAsync_NegativeTest --tests 1
 * TEST_NAMED: %t hipMemcpy2DAsync_H2D_D2H --tests 3
 * TEST_NAMED: %t hipMemcpy2DAsync_D2D --tests 4
 * TEST_NAMED: %t hipMemcpy2DAsync_WithKernel --tests 5
 * TEST_NAMED: %t hipMemcpy2DAsync_PinnedMemory --tests 6
 * HIT_END
 */

#include "test_common.h"

#define NUM_H 256
#define NUM_W 256
#define COLUMNS 8
#define ROWS 8
#define ITER 10

__global__ void
vector_square(char* B_d, char* C_d, size_t elements) {
  for (int i=0 ; i < elements ; i++) {
    C_d[i] = B_d[i] * B_d[i];
  }
}

class Memcpy2DAsync {
  char *A_h{nullptr}, *A_d{nullptr}, *B_h{nullptr},
       *B_d{nullptr}, *C_h{nullptr}, *C_d{nullptr};
  size_t pitch_A, pitch_B, pitch_C;
  size_t width{NUM_W * sizeof(char)};
  size_t sizeElements{width * NUM_H};
  size_t elements{NUM_W * NUM_H};
  hipStream_t stream;
  bool ValidateResult(char *result, int compare);
 public:
  void AllocateMemory();
  void DeAllocateMemory();
  bool Memcpy2DAsync_NegativeTest();
  bool Memcpy2DAsync_NegativeTest_SizeCheck();
  bool Memcpy2DAsync_H2D_D2HKind();
  bool Memcpy2DAsync_D2DKind_SameGPU();
  bool Memcpy2DAsync_D2DKind_MultiGPU();
  bool Memcpy2DAsync_WithKernel();
  bool Memcpy2DAsync_PinnedMemory_SameGPU();
  bool Memcpy2DAsync_PinnedMemory_MultiGPU();
};

void Memcpy2DAsync::AllocateMemory() {
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
  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&C_d),
        &pitch_C, width, NUM_H));
  for (size_t i=0; i < elements; i++) {
    A_h[i] = 3;
    B_h[i] = 4;
    C_h[i] = 123;
  }
  HIPCHECK(hipStreamCreate(&stream));
}

void Memcpy2DAsync::DeAllocateMemory() {
  HIPCHECK(hipFree(A_d)); HIPCHECK(hipFree(B_d)); HIPCHECK(hipFree(C_d));
  free(A_h); free(B_h);
  HIPCHECK(hipStreamDestroy(stream));
}

bool Memcpy2DAsync::ValidateResult(char *result, int compare) {
  for (int row = 0; row < ROWS; row++) {
    for (int column = 0; column < COLUMNS; column++) {
      if (result[(row * NUM_H) + column] != compare) {
        return false;
      }
    }
  }
  return true;
}

bool Memcpy2DAsync::Memcpy2DAsync_H2D_D2HKind() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool testResult = true;
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  // hipMemcpy Device to Host
  HIPCHECK(hipMemcpy2DAsync(A_h, width, A_d, pitch_A,
        COLUMNS, ROWS, hipMemcpyDeviceToHost, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  testResult = ValidateResult(A_h, memsetval);
  // hipMemcpy Host to Device and validating the
  // result by copying the device data to host data
  HIPCHECK(hipMemcpy2DAsync(B_d, pitch_B, B_h, width,
        COLUMNS, ROWS, hipMemcpyHostToDevice, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpy2DAsync(C_h, width, B_d, pitch_B,
        COLUMNS, ROWS, hipMemcpyDeviceToHost, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  testResult &= ValidateResult(C_h, B_h[0]);
  DeAllocateMemory();
  return testResult;
}

bool Memcpy2DAsync::Memcpy2DAsync_PinnedMemory_SameGPU() {
  HIPCHECK(hipSetDevice(0));
  bool testResult = true;
  AllocateMemory();
  char *D_h{nullptr};
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), sizeElements));
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  HIPCHECK(hipMemcpy2DAsync(D_h, width, A_d, pitch_A,
        COLUMNS, ROWS, hipMemcpyDeviceToHost, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  testResult = ValidateResult(D_h, memsetval);
  DeAllocateMemory();
  HIPCHECK(hipHostFree(D_h));
  return testResult;
}

bool Memcpy2DAsync::Memcpy2DAsync_PinnedMemory_MultiGPU() {
  bool testResult = true;
  int numDevices = 0;
  int canAccessPeer = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    hipDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    // Check for peer devices and performing D2D on the devices
    if (canAccessPeer) {
      HIPCHECK(hipSetDevice(0));
      char *D_h{nullptr};
      HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&D_h), sizeElements));
      AllocateMemory();
      HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
      HIPCHECK(hipSetDevice(1));
      hipStream_t p_stream;
      HIPCHECK(hipStreamCreate(&p_stream));
      HIPCHECK(hipMemcpy2DAsync(D_h, width, A_d, pitch_A,
               COLUMNS, ROWS, hipMemcpyDeviceToHost, p_stream));
      HIPCHECK(hipStreamSynchronize(p_stream));
      testResult = ValidateResult(D_h, memsetval);
      DeAllocateMemory();
      HIPCHECK(hipHostFree(D_h));
      HIPCHECK(hipStreamDestroy(p_stream));
    } else {
      printf("skipping the tescase as device does not have P2P");
    }
  } else {
    printf("skipped the testcase as no of devices is less than 2");
  }
  return testResult;
}

bool Memcpy2DAsync::Memcpy2DAsync_D2DKind_SameGPU() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool testResult = true;
  // Performs D2D on same GPU device
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  HIPCHECK(hipMemcpy2DAsync(B_d, pitch_B, A_d,
        pitch_A, COLUMNS, ROWS, hipMemcpyDeviceToDevice, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpy2DAsync(B_h, width, B_d, pitch_B,
        COLUMNS, ROWS, hipMemcpyDeviceToHost, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  testResult = ValidateResult(B_h, memsetval);
  DeAllocateMemory();
  return testResult;
}

bool Memcpy2DAsync::Memcpy2DAsync_D2DKind_MultiGPU() {
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
        hipStream_t p_stream;
        HIPCHECK(hipStreamCreate(&p_stream));
        char *X_d{nullptr};
        size_t pitch_X;
        HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&X_d),
              &pitch_X, width, NUM_H));
        HIPCHECK(hipMemcpy2DAsync(X_d, pitch_X, A_d,
              pitch_A, COLUMNS, ROWS, hipMemcpyDeviceToDevice, p_stream));
        HIPCHECK(hipStreamSynchronize(p_stream));
        HIPCHECK(hipMemcpy2DAsync(C_h, width, X_d,
              pitch_X, COLUMNS, ROWS, hipMemcpyDeviceToHost, p_stream));
        HIPCHECK(hipStreamSynchronize(p_stream));
        testResult &= ValidateResult(C_h, memsetval);
        HIPCHECK(hipFree(X_d));
        DeAllocateMemory();
        HIPCHECK(hipStreamDestroy(p_stream));
      } else {
        printf("Machine does not seem to have P2P between 0 & %d", j);
      }
    }
  } else {
    printf("skipped the testcase as no of devices is less than 2");
  }
  return testResult;
}

bool Memcpy2DAsync::Memcpy2DAsync_WithKernel() {
  HIPCHECK(hipSetDevice(0));
  unsigned int ThreadsperBlock = 1;
  unsigned int numBlocks = 1;
  bool testResult = true;
  AllocateMemory();
  for (int k = 0 ; k < ITER ; k++) {
    HIPCHECK(hipMemset2D(B_d, pitch_B, B_h[0], NUM_W, NUM_H));
    hipLaunchKernelGGL(vector_square, numBlocks, ThreadsperBlock, 0,
  stream, B_d, C_d, elements);
    HIPCHECK(hipMemcpy2DAsync(B_d, pitch_B, C_d, pitch_C, COLUMNS, ROWS,
    hipMemcpyDeviceToDevice, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipMemcpy2DAsync(C_h, width, B_d, pitch_B,
    COLUMNS, ROWS, hipMemcpyDeviceToHost, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    testResult &= ValidateResult(C_h, B_h[0]*B_h[0]);
  }
  DeAllocateMemory();
  return testResult;
}

bool Memcpy2DAsync::Memcpy2DAsync_NegativeTest() {
  HIPCHECK(hipSetDevice(0));
  bool TestPassed = true;
  AllocateMemory();
  hipError_t err;
  // hipMemcpy2DAsyncAsync API by Passing nullptr to Source Pointer`
  err = hipMemcpy2DAsync(A_h, width, nullptr,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost, stream);
  if (err == hipSuccess) {
    printf("hipMemcpyAsync failed when source  pointer is null");
    TestPassed = false;
  }
  // hipMemcpy2DAsyncAsync API by Passing nullptr to Destination Pointer
  err = hipMemcpy2DAsync(nullptr, width, A_d,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost, stream);
  if (err == hipSuccess) {
    printf("hipMemcpyAsync failed when dest pointer is null");
    TestPassed = false;
  }
  // hipMemcpy2DAsyncAsync API by Passing nullptr
  // to both Source and Destination ptr
  err = hipMemcpy2DAsync(nullptr, width, nullptr,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost, stream);
  if (err == hipSuccess) {
    printf("hipMemcpyAsync failed when both source and dest pointer are null");
    TestPassed = false;
  }
  // hipMemcpy2DAsyncAsync API where width is more than destination pitch
  err = hipMemcpy2DAsync(A_h, 10, A_d, pitch_A,
      NUM_W, NUM_H, hipMemcpyDeviceToHost, stream);
  if (err == hipSuccess) {
    printf("hipMemcpyAsync failed where width is more than destination pitch");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}

bool Memcpy2DAsync::Memcpy2DAsync_NegativeTest_SizeCheck() {
  HIPCHECK(hipSetDevice(0));
  AllocateMemory();
  bool TestPassed = true;
  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));
  hipError_t err;
  // hipMemcpy2DAsync API where Destination Pitch is zero
  err = hipMemcpy2DAsync(A_h, 0, A_d,
      pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost, stream);
  if (err == hipSuccess) {
    printf("hipMemcpy2DAsync failed when source pitch is null");
    TestPassed = false;
  }
  // hipMemcpy2DAsync API where Source Pitch is zero
  err = hipMemcpy2DAsync(A_h, width, A_d,
      0, NUM_W, NUM_H, hipMemcpyDeviceToHost, stream);
  if (err == hipSuccess) {
    printf("hipMemcpy2DAsync failed when destination pitch is null");
    TestPassed = false;
  }
  // hipMemcpy2DAsync API where Source and Destination Pitch are zero
  err = hipMemcpy2DAsync(A_h, 0, A_d,
      0, NUM_W, NUM_H, hipMemcpyDeviceToHost, stream);
  if (err == hipSuccess) {
    printf("hipMemcpy2DAsync failed source and destination pitches are null");
    TestPassed = false;
  }
  // hipMemcpy2DAsync API where height is zero
  // hipMemcpy2DAsync API would return success for width and height as 0
  // Validating the result with the initialized value
  err = hipMemcpy2DAsync(A_h, width, A_d,
      pitch_A, NUM_W, 0, hipMemcpyDeviceToHost, stream);
  HIPCHECK(hipStreamSynchronize(stream));
  if (err == hipSuccess) {
    TestPassed = ValidateResult(A_h, 3);
  } else {
    printf("hipMemcpy2DAsync failed when Width is null");
    TestPassed = false;
  }
  // hipMemcpy2DAsync API where width is zero
  // hipMemcpy2DAsync API would return success for width and height as 0
  // Validating the result with the initialized value
  err = hipMemcpy2DAsync(A_h, width, A_d,
      pitch_A, 0, NUM_H, hipMemcpyDeviceToHost, stream);
  HIPCHECK(hipStreamSynchronize(stream));
  if (err == hipSuccess) {
    TestPassed = ValidateResult(A_h, 3);
  } else {
    printf("hipMemcpy2DAsync failed when Width is null");
    TestPassed = false;
  }
  DeAllocateMemory();
  return TestPassed;
}

int main(int argc, char* argv[]) {
  Memcpy2DAsync Memcpy2DAsyncObj;
  HipTest::parseStandardArguments(argc, argv, false);
  bool TestPassed = true;
  if (p_tests == 1) {
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_NegativeTest();
  } else if (p_tests == 2) {
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_NegativeTest_SizeCheck();
  } else if (p_tests == 3) {
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_H2D_D2HKind();
  } else if (p_tests == 4) {
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_D2DKind_SameGPU();
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_D2DKind_MultiGPU();
  } else if (p_tests == 5) {
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_WithKernel();
  } else if (p_tests == 6) {
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_PinnedMemory_MultiGPU();
    TestPassed &= Memcpy2DAsyncObj.Memcpy2DAsync_PinnedMemory_SameGPU();
  } else {
    failed("Didnt receive any valid option. Try options 1 to 6\n");
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Failed!");
  }
}
