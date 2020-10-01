/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */



// Test Description:
// This test case verifies the working of hipFuncSetSharedMemConfig() api and
// the flag parameter

#include "test_common.h"


__global__ void ReverseSeq(int *A, int *B, int N) {
  extern __shared__ int SMem[];
  int offset = threadIdx.x;
  int MirrorVal = N - offset - 1;
  SMem[offset] = A[offset];
  __syncthreads();
  B[offset] = SMem[MirrorVal];
}

int main() {
  bool IfTestPassed = true;
  int *Ah = NULL, *RAh = NULL, NELMTS = 128;
  int *Ad = NULL, *RAd = NULL;
  Ah = reinterpret_cast<int*>(malloc(NELMTS * sizeof(int)));
  RAh = reinterpret_cast<int*>(malloc(NELMTS * sizeof(int)));
  HIPCHECK(hipMalloc(&Ad, NELMTS * sizeof(int)));
  HIPCHECK(hipMalloc(&RAd, NELMTS * sizeof(int)));
  for (int i = 0; i < NELMTS; ++i) {
    Ah[i] = i;
    RAh[i] = NELMTS - i - 1;
  }
  HIPCHECK(hipMemcpy(Ad, Ah, NELMTS * sizeof(int), hipMemcpyHostToDevice));
  HIPCHECK(hipMemset(RAd, 0, NELMTS * sizeof(int)));
  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeDefault flag
  HIPCHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(&ReverseSeq),
                                     hipSharedMemBankSizeDefault));
  // Kernel Launch with shared mem size of = NELMTS * sizeof(int)
  ReverseSeq<<<1, NELMTS, NELMTS * sizeof(int)>>>(Ad, RAd, NELMTS);
  memset(Ah, 0, NELMTS * sizeof(int));
  // Verifying the results
  HIPCHECK(hipMemcpy(Ah, RAd, NELMTS * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < NELMTS; ++i) {
    if (Ah[i] != RAh[i]) {
      printf("Mismatch found at %d value of array\n", i);
      printf(" after setting the flag hipSharedMemBankSizeDefault\n");
      IfTestPassed = false;
    }
  }
  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeFourBytes flg
  HIPCHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(&ReverseSeq),
                                     hipSharedMemBankSizeFourByte));
  HIPCHECK(hipMemset(RAd, 0, NELMTS * sizeof(int)));
  // Kernel Launch with shared mem size of = NELMTS * sizeof(int)
  ReverseSeq<<<1, NELMTS, NELMTS * sizeof(int)>>>(Ad, RAd, NELMTS);
  memset(Ah, 0, NELMTS * sizeof(int));
  // Verifying the results
  HIPCHECK(hipMemcpy(Ah, RAd, NELMTS * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < NELMTS; ++i) {
    if (Ah[i] != RAh[i]) {
      printf("Mismatch found at %d value of array\n", i);
      printf(" after setting the flag hipSharedMemBankSizeFourByte\n");
      IfTestPassed = false;
    }
  }
  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeEightBytes flg
  HIPCHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(&ReverseSeq),
                                      hipSharedMemBankSizeEightByte));
  HIPCHECK(hipMemset(RAd, 0, NELMTS * sizeof(int)));
  // Kernel Launch with shared mem size of = NELMTS * sizeof(int)
  ReverseSeq<<<1, NELMTS, NELMTS * sizeof(int)>>>(Ad, RAd, NELMTS);
  memset(Ah, 0, NELMTS * sizeof(int));
  // Verifying the results
  HIPCHECK(hipMemcpy(Ah, RAd, NELMTS * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < NELMTS; ++i) {
    if (Ah[i] != RAh[i]) {
      printf("Mismatch found at %d value of array\n", i);
      printf(" after setting the flag hipSharedMemBankSizeEightByte\n");
      IfTestPassed = false;
    }
  }

  free(Ah);
  free(RAh);
  HIPCHECK(hipFree(Ad));
  HIPCHECK(hipFree(RAd));

  if (IfTestPassed) {
    passed();
  } else {
    failed("");
  }
}
