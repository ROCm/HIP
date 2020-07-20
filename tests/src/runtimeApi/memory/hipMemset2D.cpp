/*
 * Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Test for hipMemset2D functionality for different width and height values

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t hipMemset2D-basic
 * TEST_NAMED: %t hipMemset2D-dim1 --width2D 10 --height2D 10 --memsetWidth 4 --memsetHeight 4
 * TEST_NAMED: %t hipMemset2D-dim2 --width2D 100 --height2D 100 --memsetWidth 20 --memsetHeight 40
 * TEST_NAMED: %t hipMemset2D-dim3 --width2D 256 --height2D 256 --memsetWidth 39 --memsetHeight 19
 * TEST_NAMED: %t hipMemset2D-zeroH --width2D 100 --height2D 100 --memsetWidth 20 --memsetHeight 0
 * TEST_NAMED: %t hipMemset2D-zeroW --width2D 100 --height2D 100 --memsetWidth 0 --memsetHeight 20
 * TEST_NAMED: %t hipMemset2D-zeroW*H --width2D 100 --height2D 100 --memsetWidth 0 --memsetHeight 0
 * HIT_END
 */

#include "test_common.h"

// Check hipMemset2D functionality
bool testhipMemset2D(int memsetval, int p_gpuDevice) {
  bool testResult = true;
  size_t numH = 256;
  size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH;
  size_t elements = numW* numH;
  printf("testhipMemset2D memsetval=%2x device=%d\n", memsetval, p_gpuDevice);
  char *A_d;
  char *A_h;

  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width ,
                          numH));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);

  for (size_t i=0; i < elements; i++) {
    A_h[i] = 1;
  }

  HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, numW, numH));
  HIPCHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH,
                       hipMemcpyDeviceToHost));

  for (int i=0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      testResult = false;
      printf("testhipMemset2D mismatch at index:%d computed:%02x, memsetval:"
             "%02x\n", i, static_cast<int>(A_h[i]), static_cast<int>(memsetval));
      break;
    }
  }

  hipFree(A_d);
  free(A_h);
  return testResult;
}

// Check hipMemset2DAsync functionality
bool testhipMemset2DAsync(int memsetval, int p_gpuDevice) {
  size_t numH = 256;
  size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH;
  size_t elements = numW * numH;
  printf("testhipMemset2DAsync memsetval=%2x device=%d\n", memsetval,
          p_gpuDevice);
  char *A_d;
  char *A_h;
  bool testResult = true;

  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                          width , numH));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);

  for (size_t i=0; i < elements; i++) {
      A_h[i] = 1;
  }

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  HIPCHECK(hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, stream));
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH,
                       hipMemcpyDeviceToHost));

  for (int i=0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      testResult = false;
      printf("testhipMemset2DAsync mismatch at index:%d computed:%02x, memsetval:"
             "%02x\n", i, static_cast<int>(A_h[i]), static_cast<int>(memsetval));
      break;
    }
  }

  hipFree(A_d);
  HIPCHECK(hipStreamDestroy(stream));
  free(A_h);
  return testResult;
}

int width2D = 20;
int height2D = 20;
int memsetWidth = 20;
int memsetHeight = 20;

int parseExtraArguments(int argc, char* argv[]) {
  int i = 0;
  for (i = 1; i < argc; i++) {
    const char* arg = argv[i];
    if (!strcmp(arg, " ")) {
      // skip NULL args.
    } else if (!strcmp(arg, "--width2D")) {
        if (++i >= argc || !HipTest::parseInt(argv[i], &width2D)) {
          failed("Bad width2D argument");
        }
    } else if (!strcmp(arg, "--height2D")) {
        if (++i >= argc || !HipTest::parseInt(argv[i], &height2D)) {
          failed("Bad height2D argument");
        }
    } else if (!strcmp(arg, "--memsetWidth")) {
        if (++i >= argc || !HipTest::parseInt(argv[i], &memsetWidth)) {
          failed("Bad memsetWidth argument");
        }
    } else if (!strcmp(arg, "--memsetHeight")) {
        if (++i >= argc || !HipTest::parseInt(argv[i], &memsetHeight)) {
          failed("Bad memsetHeight argument");
        }
    } else {
        failed("Bad argument");
    }
  }
  return i;
}

// Memset random dimensions
bool testMemset2DPartial(int memsetval, int p_gpuDevice) {
  bool testResult = true;
  size_t NUM_H = height2D;
  size_t NUM_W = width2D;
  size_t Nbytes = N*sizeof(char);
  size_t pitch_A;
  size_t width = NUM_W * sizeof(char);
  size_t sizeElements = width * NUM_H;
  size_t elements = NUM_W * NUM_H;
  char *A_d;
  char *A_h;
  printf("testhipMemset2DPartial memsetval=%2x device=%d\n", memsetval,
          p_gpuDevice);

  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                          width, NUM_H));
  hipError_t e;
  int index;

  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);

  for (index = 0; index < sizeElements; index++) {
    A_h[0] = 'c';
  }

  printf("2D Dimension: %zuX%zu, MemsetWidth:%d, memsetHeight:%d\n",
         NUM_W, NUM_H, memsetWidth, memsetHeight);
  e = hipMemset2D(A_d, pitch_A, memsetval, memsetWidth, memsetHeight);
  HIPASSERT(e == hipSuccess);

  HIPCHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, NUM_W, NUM_H,
                       hipMemcpyDeviceToHost));

  for (int row = 0; row < memsetHeight; row++) {
    for (int column = 0; column < memsetWidth; column++) {
      if (A_h[(row * width) + column] != memsetval) {
        printf("A_h[%d][%d] did not match %d", row, column, memsetval);
        testResult = false;
      }
    }
  }
  hipFree(A_d);
  free(A_h);
  return testResult;
}

int main(int argc, char *argv[]) {
  int extraArgs = 0;
  bool testResult = true;

  HIPCHECK(hipSetDevice(p_gpuDevice));
  extraArgs = HipTest::parseStandardArguments(argc, argv, false);
  parseExtraArguments(extraArgs, argv);

  if (extraArgs == 1) {
    testResult &= testhipMemset2D(memsetval, p_gpuDevice);
    if (!(testResult)) {
      printf("hipMemset2D failed\n");
    }
    testResult &= testhipMemset2DAsync(memsetval, p_gpuDevice);
    if (!(testResult)) {
      printf("hipMemset2DAsync failed\n");
    }
  } else if (extraArgs == 9) {
      testResult &= testMemset2DPartial(memsetval, p_gpuDevice);
      if (!(testResult)) {
        printf("hipMemset2D at random dimensions failed\n");
      }
  } else {
      failed("Wrong Arguments for test\n");
  }

  if (testResult) {
    passed();
  } else {
      failed("one or more hipMemset2D tests failed");
  }
}
