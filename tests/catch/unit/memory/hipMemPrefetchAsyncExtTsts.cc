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

#include <hip_test_common.hh>
// Kernel function
__global__ void MemPrftchAsyncKernel(int* Hmm, const int* Hmm1, size_t N) {
  size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
  size_t stride = hipBlockDim_x * hipGridDim_x;
  for (size_t i = offset; i < N; i += stride) {
    Hmm[i] = Hmm1[i] * Hmm1[i];
  }
}

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

/* Test Case Description: Allocate managed memory --> prefetch to gpu 0
   call hipMemAdvise() on the memory and apply the flags ReadMostly,
   AccessedBy, and PreferredLocation for gpus other than gpu 0 and verify
   the flags using hipMemGetAttribute()*/
TEST_CASE("Unit_hipMemPrefetchAsyncAdviseFlgTst") {
    int NGpus = 0;
    HIP_CHECK(hipGetDeviceCount(&NGpus));
    if (NGpus >= 2) {
      int MangdMem = HmmAttrPrint();
      if (MangdMem == 1) {
        int *Hmm = nullptr, MemSz = (4096 * 4), InitVal = 123;
        int Outpt = 9999, NumElms = MemSz/4;
        bool IfTestPassed = true;
        hipStream_t strm;
        HIP_CHECK(hipStreamCreate(&strm));
        HIP_CHECK(hipMallocManaged(&Hmm, MemSz));
        // Initializing the memory
        for (int i = 0; i < NumElms; ++i) {
          Hmm[i] = InitVal;
        }
        HIP_CHECK(hipMemPrefetchAsync(Hmm, MemSz, 0, strm));
        HIP_CHECK(hipStreamSynchronize(strm));
        HIP_CHECK(hipMemAdvise(Hmm, MemSz, hipMemAdviseSetReadMostly, 1));
        HIP_CHECK(hipMemRangeGetAttribute(&Outpt, sizeof(int),
                  hipMemRangeAttributeReadMostly, Hmm, MemSz));
        if (Outpt != 1) {
          WARN("hipMemRangeAttributeReadMostly flag did not take effect"
               " as expected!!");
          IfTestPassed = false;
        }
        HIP_CHECK(hipMemAdvise(Hmm, MemSz, hipMemAdviseSetAccessedBy, 1));
        HIP_CHECK(hipMemRangeGetAttribute(&Outpt, sizeof(int),
                  hipMemRangeAttributeAccessedBy, Hmm, MemSz));
        if (Outpt != 1) {
          WARN("hipMemRangeAttributeAccessedBy flag did not take effect"
               " as expected!!");
          IfTestPassed = false;
        }
        HIP_CHECK(hipMemAdvise(Hmm, MemSz, hipMemAdviseSetPreferredLocation,
                               1));
        HIP_CHECK(hipMemRangeGetAttribute(&Outpt, sizeof(int),
                  hipMemRangeAttributePreferredLocation, Hmm, MemSz));
        if (Outpt != 1) {
          WARN("hipMemRangeAttributePreferredLocation flag did not take effect"
               " as expected!!");
          IfTestPassed = false;
        }
        HIP_CHECK(hipStreamDestroy(strm));
        HIP_CHECK(hipFree(Hmm));
        REQUIRE(IfTestPassed);
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
              "attribute. Hence skipping the testing with Pass result.\n");
    }
  } else {
    SUCCEED("This test needs atleast 2 gpus, but the system has less than"
            " 2 gpus hence skipping the test");
  }
}

/* Test Case description: Allocate managed memory --> set AccessedBy using
    hipMemAdvise() to gpu1 prefetch the memory to gpu 0 and then query for
    AccessedBy using hipMemGetAttribute() and validate if AccessedBy is still
    set to gpu1. Similar tests are done with ReadMostly and PreferredLocation
    flags */
TEST_CASE("Unit_hipMemPrefetchAsyncAccsdByTst") {
  int NGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&NGpus));
  if (NGpus >= 2) {
    int MangdMem = HmmAttrPrint();
    if (MangdMem == 1) {
      int *Hmm = nullptr, MemSz = (4096 * 4), InitVal = 123, NumElms = MemSz/4;
      int Outpt = 9999;
      bool IfTestPassed = true;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMallocManaged(&Hmm, MemSz));
      // Initializing the memory
      for (int i = 0; i < NumElms; ++i) {
        Hmm[i] = InitVal;
      }
      SECTION("Test AccessedBy with Prefetch") {
        HIP_CHECK(hipMemAdvise(Hmm, MemSz, hipMemAdviseSetAccessedBy, 1));
        HIP_CHECK(hipMemPrefetchAsync(Hmm, MemSz, 0, strm));
        HIP_CHECK(hipStreamSynchronize(strm));
        HIP_CHECK(hipMemRangeGetAttribute(&Outpt, sizeof(int),
                    hipMemRangeAttributeAccessedBy, Hmm, MemSz));
        if (Outpt != 1) {
          WARN("hipMemRangeAttributeAccessedBy flag did not take effect"
               " as expected!!");
          IfTestPassed = false;
        }
      }
      SECTION("Test ReadMostly with Prefetch") {
        HIP_CHECK(hipMemAdvise(Hmm, MemSz, hipMemAdviseSetReadMostly, 1));
        HIP_CHECK(hipMemPrefetchAsync(Hmm, MemSz, 0, strm));
        HIP_CHECK(hipStreamSynchronize(strm));
        MemPrftchAsyncKernel1<<<(NumElms/32), 32, 0, strm>>>(Hmm, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        HIP_CHECK(hipMemRangeGetAttribute(&Outpt, sizeof(int),
                    hipMemRangeAttributeReadMostly, Hmm, MemSz));
        if (Outpt != 1) {
          WARN("hipMemRangeAttributeReadMostly flag did not take effect"
               " as expected!!");
          IfTestPassed = false;
        }
        // Verifying the results
        for (int i = 0; i < NumElms; ++i) {
          if (Hmm[i] != (InitVal * InitVal)) {
            WARN("Did not receive expected value!!");
            IfTestPassed = false;
            break;
          }
        }
      }
      SECTION("Test PreferredLocation with Prefetch") {
        HIP_CHECK(hipMemAdvise(Hmm, MemSz, hipMemAdviseSetPreferredLocation,
                               1));
        HIP_CHECK(hipMemPrefetchAsync(Hmm, MemSz, 0, strm));
        HIP_CHECK(hipStreamSynchronize(strm));
        MemPrftchAsyncKernel1<<<(NumElms/32), 32, 0, strm>>>(Hmm, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        HIP_CHECK(hipMemRangeGetAttribute(&Outpt, sizeof(int),
                    hipMemRangeAttributePreferredLocation, Hmm, MemSz));
        if (Outpt != 1) {
          WARN("hipMemRangeAttributePreferredLocation flag did not take effect"
             " as expected!!");
          IfTestPassed = false;
        }
        // Verifying the results
        for (int i = 0; i < NumElms; ++i) {
          if (Hmm[i] != (InitVal * InitVal)) {
            WARN("Did not receive expected value!!");
            IfTestPassed = false;
            break;
          }
        }
      }
      HIP_CHECK(hipFree(Hmm));
      HIP_CHECK(hipStreamDestroy(strm));
      REQUIRE(IfTestPassed);
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the testing with Pass result.\n");
    }
  } else {
    SUCCEED("This test needs atleast 2 gpus, but the system has less than"
            " 2 gpus hence skipping the test");
  }
}

/*Test Case description: Negative testing with hipMemPrefetchAsync() api*/
TEST_CASE("Unit_hipMemPrefetchAsyncNegativeTst") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    int *Hmm = nullptr, MemSz = 4096*4, NumElms = MemSz/4, InitVal = 123;
    bool IfTestPassed = true;
    HIP_CHECK(hipMallocManaged(&Hmm, MemSz));
    for (int i = 0; i < NumElms; ++i) {
      Hmm[i] = InitVal;
    }
    hipError_t err;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    SECTION("Passing null for dev ptr") {
      int *Ptr;
      err = hipMemPrefetchAsync(NULL, MemSz, 0, strm);
      if (err == hipSuccess) {
        WARN("hipMemPrefetchAsync() gives hipSuccess when NULL is passed!!");
        IfTestPassed = false;
      }
      err = hipMemPrefetchAsync(Ptr, MemSz, 0, strm);
      if (err == hipSuccess) {
        WARN("hipMemPrefetchAsync() gives hipSuccess when uninitialized"
             " pointer is passed!!");
        IfTestPassed = false;
      }
    }

    SECTION("Passing unusual count size(2nd param)") {
      // Passing count size as zero
      // expectation: Api should return error
      err = hipMemPrefetchAsync(Hmm, 0, 0, strm);
      if (err == hipSuccess) {
        WARN("hipMemPrefetchAsync() gives hipSuccess when count size is"
             " passed as zero!!");
        IfTestPassed = false;
      }
      // Passing count size half of actually allocated
      // expectation: No issue should be observed
      err = hipMemPrefetchAsync(Hmm, MemSz/2, 0, strm);
      if (err != hipSuccess) {
        WARN("hipMemPrefetchAsync() returned error when count size passed is"
             " half of actually allocated!!");
        IfTestPassed = false;
      }
      // Passing count size double that of actually allocated
      // expectation: Api should return error
      err = hipMemPrefetchAsync(Hmm, MemSz*2, 0, strm);
      if (err == hipSuccess) {
        WARN("hipMemPrefetchAsync() gives hipSuccess when count size passed is"
             " double that of actually allocated!!");
        IfTestPassed = false;
      }
    }
    SECTION("Passing invalid device Ordinal") {
      err = hipMemPrefetchAsync(Hmm, MemSz, 9999, strm);
      if (err == hipSuccess) {
        WARN("hipMemPrefetchAsync() gives hipSuccess when Invalid device"
             " ordinal is passed!!");
        IfTestPassed = false;
      }
    }
    SECTION("Checking behavior with stream object") {
      // Passing Null stream
      // expectation: No issue should be observed
      err = hipMemPrefetchAsync(Hmm, MemSz, 0, 0);
      if (err != hipSuccess) {
        WARN("hipMemPrefetchAsync() returns error when Null stream is"
             "passed!!");
        IfTestPassed = false;
      }
    // Passing stream object belong to destination device
    // expectation: No issue should be observed
    int NGpus = 0;
    HIP_CHECK(hipGetDeviceCount(&NGpus));
    if (NGpus > 1) {
      hipStream_t strm1;
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipStreamCreate(&strm1));
      err = hipMemPrefetchAsync(Hmm, MemSz, 1, strm1);
      if (err != hipSuccess) {
        WARN("hipMemPrefetchAsync() returns error when stream object"
             " created in the context of destination gpu is passed!!");
        IfTestPassed = false;
      }
      HIP_CHECK(hipStreamDestroy(strm1));
    }
  }
  HIP_CHECK(hipFree(Hmm));
  HIP_CHECK(hipStreamDestroy(strm));
  REQUIRE(IfTestPassed);

  } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* Test Case description: In this test case I am trying to allocate HMM memory
   which is not multiple of page Size, but still trying to launch kernel and
   see if we are getting values as expected.*/
TEST_CASE("Unit_hipMemPrefetchAsync_NonPageSz") {
  int *Hmm = nullptr, NumElms = 4096*2, InitVal = 123;
  hipStream_t strm;
  bool IfTestPassed = true;
  HIP_CHECK(hipStreamCreate(&strm));
  // Allocating memory = 2*Page Size + 8 bytes
  HIP_CHECK(hipMallocManaged(&Hmm, (NumElms * sizeof(int) + 8)));
  for (int i = 0; i < (NumElms + 2); ++i) {
    Hmm[i] = InitVal;
  }
  HIP_CHECK(hipMemPrefetchAsync(Hmm, (NumElms * sizeof(int) + 8), 0, strm));
  HIP_CHECK(hipStreamSynchronize(strm));
  MemPrftchAsyncKernel1<<<((NumElms + 2)/32 + 1), 32>>>(Hmm, (NumElms + 2));
  HIP_CHECK(hipStreamSynchronize(strm));
  for (int i = 0; i < (NumElms + 2); ++i) {
    if (Hmm[i] != (InitVal * InitVal)) {
      WARN("Didnt receive expected output after kernel launch!!");
      IfTestPassed = false;
      break;
    }
  }
  HIP_CHECK(hipFree(Hmm));
  HIP_CHECK(hipStreamDestroy(strm));
  REQUIRE(IfTestPassed);
}
