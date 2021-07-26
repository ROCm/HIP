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

// Test Description:
// This test case verifies the working of hipFuncSetSharedMemConfig() api and
// the flag parameter

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>


__global__ void ReverseSeq(int *A, int *B, int N) {
  extern __shared__ int SMem[];
  int offset = threadIdx.x;
  int MirrorVal = N - offset - 1;
  SMem[offset] = A[offset];
  __syncthreads();
  B[offset] = SMem[MirrorVal];
}
/*
This testcase verifies the basic functionality of hipFuncSetSharedMemConfig API
by setting shared memory bank size

1. hipSharedMemBankSizeDefault
2. hipSharedMemBankSizeFourByte
3. hipSharedMemBankSizeEightByte

*/
TEST_CASE("Unit_hipFuncSetSharedMemConfig_Basic") {
  int *Ah{nullptr}, *RAh{nullptr}, NumElms = 128;
  int *Ad{nullptr}, *RAd{nullptr};

  HipTest::initArrays<int>(&Ad, &RAd, nullptr,
                           &Ah, &RAh, nullptr, NumElms, false);
  for (int i = 0; i < NumElms; ++i) {
    Ah[i] = i;
    RAh[i] = NumElms - i - 1;
  }
  HIP_CHECK(hipMemcpy(Ad, Ah, NumElms * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(RAd, 0, NumElms * sizeof(int)));

  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeDefault flag
  HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>
                                      (&ReverseSeq),
                                      hipSharedMemBankSizeDefault));

  // Kernel Launch with shared mem size of = NumElms * sizeof(int)
  ReverseSeq<<<1, NumElms, NumElms * sizeof(int)>>>(Ad, RAd, NumElms);
  memset(Ah, 0, NumElms * sizeof(int));

  // Verifying the results
  HIP_CHECK(hipMemcpy(Ah, RAd, NumElms * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < NumElms; ++i) {
    REQUIRE(Ah[i] == RAh[i]);
  }

  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeFourBytes flg
  HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>
                                      (&ReverseSeq),
                                      hipSharedMemBankSizeFourByte));
  HIP_CHECK(hipMemset(RAd, 0, NumElms * sizeof(int)));

  // Kernel Launch with shared mem size of = NumElms * sizeof(int)
  ReverseSeq<<<1, NumElms, NumElms * sizeof(int)>>>(Ad, RAd, NumElms);
  memset(Ah, 0, NumElms * sizeof(int));

  // Verifying the results
  HIP_CHECK(hipMemcpy(Ah, RAd, NumElms * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < NumElms; ++i) {
    REQUIRE(Ah[i] == RAh[i]);
  }

  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeEightBytes flg
  HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>
                                      (&ReverseSeq),
                                      hipSharedMemBankSizeEightByte));
  HIP_CHECK(hipMemset(RAd, 0, NumElms * sizeof(int)));

  // Kernel Launch with shared mem size of = NumElms * sizeof(int)
  ReverseSeq<<<1, NumElms, NumElms * sizeof(int)>>>(Ad, RAd, NumElms);
  memset(Ah, 0, NumElms * sizeof(int));

  // Verifying the results
  HIP_CHECK(hipMemcpy(Ah, RAd, NumElms * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < NumElms; ++i) {
    REQUIRE(Ah[i] == RAh[i]);
  }

  HipTest::freeArrays<int>(Ad, RAd, nullptr,
                            Ah, RAh, nullptr, false);
}
