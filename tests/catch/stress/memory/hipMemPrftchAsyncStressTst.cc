/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

/* Test Case Description:
   The following test allocates a managed memory and prefetch it in
   one-to-all and all-to-one fashion followed by kernel launch within available
   devices*/

#include <hip_test_common.hh>

// Kernel function
__global__ void MemPrftchAsyncKernel1(int* Hmm, size_t N) {
  size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
  size_t stride = hipBlockDim_x * hipGridDim_x;
  for (size_t i = offset; i < N; i += stride) {
    Hmm[i] = Hmm[i] * Hmm[i];
  }
}

static int HmmAttrPrint() {
  int managed = 0;
  WARN("The following are the attribute values related to HMM for"
         " device 0:\n");
  HIP_CHECK(hipDeviceGetAttribute(&managed,
              hipDeviceAttributeDirectManagedMemAccessFromHost, 0));
  WARN("hipDeviceAttributeDirectManagedMemAccessFromHost: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
                                 hipDeviceAttributeConcurrentManagedAccess, 0));
  WARN("hipDeviceAttributeConcurrentManagedAccess: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  WARN("hipDeviceAttributePageableMemoryAccess: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
              hipDeviceAttributePageableMemoryAccessUsesHostPageTables, 0));
  WARN("hipDeviceAttributePageableMemoryAccessUsesHostPageTables:"
         << managed);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  WARN("hipDeviceAttributeManagedMemory: " << managed);
  return managed;
}

static void ReleaseResource(int *Hmm, int *Hmm1, hipStream_t *strm) {
  HIP_CHECK(hipFree(Hmm));
  HIP_CHECK(hipFree(Hmm1));
  HIP_CHECK(hipStreamDestroy(*strm));
}


/* The following test allocates a managed memory and prefetch it in
   one-to-all and all-to-one fahsion followed by kernel launch within available
   devices*/
TEST_CASE("Unit_hipMemPrefetchAsyncOneToAll") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    int *Hmm = nullptr, *Hmm1 = nullptr, NumDevs, MemSz = (4096 * 4);
    int InitVal = 123, NumElms = MemSz/4;
    bool IfTestPassed = true;
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    HIP_CHECK(hipMallocManaged(&Hmm, MemSz));
    HIP_CHECK(hipMallocManaged(&Hmm1, MemSz));
    for (int i = 0; i < NumElms; ++i) {
      Hmm1[i] = InitVal;
    }
    hipStream_t strm;
    for (int i = -1; i < NumDevs; ++i) {
      HIP_CHECK(hipMemPrefetchAsync(Hmm1, MemSz, i, 0));
      for (int j = -1; j < NumDevs; ++j) {
        if (i == j) {
          continue;
        }
        if (j != -1) {
          HIP_CHECK(hipSetDevice(j));
        }
        HIP_CHECK(hipStreamCreate(&strm));
        // Prefetching memory from i to j
        HIP_CHECK(hipMemPrefetchAsync(Hmm1, MemSz, j, strm));
        HIP_CHECK(hipStreamSynchronize(strm));
        MemPrftchAsyncKernel<<<(NumElms/32), 32, 0, strm>>>(Hmm, Hmm1, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        // Verifying the result
        for (int m = 0; m < NumElms; ++m) {
          if (Hmm[m] != (InitVal * InitVal)) {
            IfTestPassed = false;
          }
        }
        if (!IfTestPassed) {
          ReleaseResource(Hmm, Hmm1, &strm);
          INFO("Did not find expected value!");
          REQUIRE(false);
        }
        // Resetting the values in Hmm
        HIP_CHECK(hipMemset(Hmm, 0, MemSz));
        // Prefetching memory from j to i
        HIP_CHECK(hipMemPrefetchAsync(Hmm1, MemSz, i, strm));
        HIP_CHECK(hipStreamSynchronize(strm));
        MemPrftchAsyncKernel<<<(NumElms/32), 32, 0, strm>>>(Hmm, Hmm1, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        // Verifying the result
        for (int m = 0; m < NumElms; ++m) {
          if (Hmm[m] != (InitVal * InitVal)) {
            IfTestPassed = false;
          }
        }
        if (!IfTestPassed) {
          ReleaseResource(Hmm, Hmm1, &strm);
          INFO("Did not find expected value!");
          REQUIRE(false);
        }
        // Resetting the values in Hmm
        HIP_CHECK(hipMemset(Hmm, 0, MemSz));
        HIP_CHECK(hipStreamDestroy(strm));
      }
    }
    // Releasing the resources in case all the scenarios passed
    HIP_CHECK(hipFree(Hmm));
    HIP_CHECK(hipFree(Hmm1));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
