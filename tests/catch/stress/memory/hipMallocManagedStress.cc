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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

//  The following test case allocation, host access, device access of HMM
//   memory from size 1 to 10KB

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define INCRMNT 10
// Kernel function
__global__ void KrnlWth2MemTypesC(unsigned char *Hmm, unsigned char *Dptr,
                                  size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Dptr[i] + INCRMNT;
  }
}
static bool IfTestPassed = true;

static int HmmAttrPrint() {
  int managed = 0;
  INFO("The following are the attribute values related to HMM for"
         " device 0:\n");
  HIP_CHECK(hipDeviceGetAttribute(&managed,
              hipDeviceAttributeDirectManagedMemAccessFromHost, 0));
  INFO("hipDeviceAttributeDirectManagedMemAccessFromHost: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
                                 hipDeviceAttributeConcurrentManagedAccess, 0));
  INFO("hipDeviceAttributeConcurrentManagedAccess: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed,
              hipDeviceAttributePageableMemoryAccessUsesHostPageTables, 0));
  INFO("hipDeviceAttributePageableMemoryAccessUsesHostPageTables:"
         << managed);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  return managed;
}
//  The following test case allocation, host access, device access of HMM
//   memory from size 1 to 10KB

TEST_CASE("Unit_hipMallocManaged_MultiSize") {
  IfTestPassed = true;
  int managed = HmmAttrPrint();
  if (managed == 1) {
    unsigned char *Hmm1 = nullptr, *Hmm2 = nullptr;
    int InitVal = 100, blockSize = 64, DataMismatch = 0;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    dim3 dimBlock(blockSize, 1, 1);
    for (int i = 1; i < (1024*1024); ++i) {
      HIP_CHECK(hipMallocManaged(&Hmm1, i));
      HIP_CHECK(hipMallocManaged(&Hmm2, i));
      for (int j = 0; j < i; ++j) {
        Hmm1[j] = InitVal;
      }
      dim3 dimGrid((i + blockSize -1)/blockSize, 1, 1);
      KrnlWth2MemTypesC<<<dimGrid, dimBlock, 0, strm>>>(Hmm2, Hmm1, i);
      HIP_CHECK(hipStreamSynchronize(strm));
      //  Verifying the results
      for (int k = 0; k < i; ++k) {
        if (Hmm2[k] != (InitVal + INCRMNT)) {
          DataMismatch++;
        }
      }
      if (DataMismatch != 0) {
        WARN("DataMismatch observed!\n");
        IfTestPassed = false;
      }
      DataMismatch = 0;
      HIP_CHECK(hipFree(Hmm1));
      HIP_CHECK(hipFree(Hmm2));
      if (IfTestPassed == false) {
        HIP_CHECK(hipStreamDestroy(strm));
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipStreamDestroy(strm));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

