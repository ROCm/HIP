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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* Test Case Description:
   Scenario-1: The following function tests the count parameter(last param) to
   hipMemRangeGetAttribute api by passing possible extreme values.
   Curently the only way to test if count param working properly is to verify
   the first parameter of hipMemRangeGetAttribute() api has value 1 stored

   Scenario-2: This test case checks the behavior of hipMemRangeGetAttribute()  with
   AccessedBy flag is consistent with cuda's counter part

   Scenario-3: Allocate  4 * page size of memory with the flag hipMemAttachGloal. Advise
   AccessedBy, ReadMostly and PreferredLocation to first half(2*pageSz) of the
   memory and probe the for the flags which are set earlier using
   hipMemRangeGetAttribute() api for the full size(4*PageSz).


   Scenario-4: The following scenarios tests that probing the attributes which are not set
   by hipMemAdvise() but being probed using hipMemRangeGetAttribute() should
   not result in a crash

   Scenario-5: The following scenario is a simple test which does the following:
   Allocate Hmm memory --> hipMemPrefetchAsync() to device 0 and then
   probe LastPrefetchLocation attribute using hipMemRangeGetAttribute

   Scenario-6: The following Test Case does negative tests on hipMemRangeGetAttribute()*/

#include <hip_test_common.hh>
#include <stdlib.h>
#ifdef __linux__
  #include <unistd.h>
  #include <sys/sysinfo.h>
#endif

static bool CheckError(hipError_t err, int LineNo) {
  if (err == hipSuccess) {
    WARN("Error expected but received hipSuccess at line no.:"
           << LineNo);
    return false;
  } else {
    return true;
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

// The following function tests the count parameter(last param) to
// hipMemRangeGetAttribute api by passing possible extreme values.
// Curently the only way to test if count param working properly is to verify
// the first parameter of hipMemRangeGetAttribute() api has value 1 stored
TEST_CASE("Unit_hipMemRangeGetAttribute_TstCountParam") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    int MEM_SIZE = 4096, RND_NUM = 9999, FLG_READMOSTLY_ENBLD = 1;
    bool IfTestPassed = true;
    int data = RND_NUM, *devPtr = nullptr;
    size_t TotGpuMem, TotGpuFreeMem;
    HIP_CHECK(hipMemGetInfo(&TotGpuFreeMem, &TotGpuMem));

    HIP_CHECK(hipMallocManaged(&devPtr, MEM_SIZE, hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(devPtr, MEM_SIZE, hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemRangeGetAttribute(reinterpret_cast<void*>(&data),
                                     sizeof(int),
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, MEM_SIZE));
    if (data != FLG_READMOSTLY_ENBLD) {
      WARN("hipMemRangeGetAttribute() api didnt return expected value!\n");
      IfTestPassed = false;
    }
    HIP_CHECK(hipFree(devPtr));
    HIP_CHECK(hipMallocManaged(&devPtr, TotGpuFreeMem, hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(devPtr, TotGpuFreeMem, hipMemAdviseSetReadMostly,
                           0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, TotGpuFreeMem));

    if (data != FLG_READMOSTLY_ENBLD) {
      WARN("hipMemRangeGetAttribute() api didnt return expected value!\n");
      IfTestPassed = false;
    }
    HIP_CHECK(hipFree(devPtr));
    HIP_CHECK(hipMallocManaged(&devPtr, (TotGpuFreeMem - 1),
                              hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(devPtr, (TotGpuFreeMem - 1),
                          hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, (TotGpuFreeMem - 1)));

    if (data != FLG_READMOSTLY_ENBLD) {
      WARN("hipMemRangeGetAttribute() api didnt return expected value!\n");
      IfTestPassed = false;
    }
    HIP_CHECK(hipFree(devPtr));

    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/* The following Test Case does negative tests on hipMemRangeGetAttribute()*/

TEST_CASE("Unit_hipMemRangeGetAttribute_NegativeTests") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    int MEM_SIZE = 4096, RND_NUM = 9999;
    float *devPtr = nullptr;
    int NumDevs;
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    int data = RND_NUM;
    int *OutData = new int[NumDevs];
    for (int m = 0; m < NumDevs; ++m) {
      OutData[m] = RND_NUM;
    }
    HIP_CHECK(hipMallocManaged(&devPtr, MEM_SIZE, hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(devPtr, MEM_SIZE, hipMemAdviseSetReadMostly, 0));

    // checking the behavior with dataSize 0
    SECTION("checking the behavior with dataSize 0") {
      REQUIRE(CheckError(hipMemRangeGetAttribute(&data, 0,
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, MEM_SIZE), __LINE__));
    }
    // checking the behavior with dataSize > 4 and even
    SECTION("checking the behavior with dataSize > 4 and even") {
      REQUIRE(CheckError(hipMemRangeGetAttribute(OutData, 6,
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, MEM_SIZE), __LINE__));
    }
    // checking the behavior with dataSize > 4 and odd
    SECTION("checking the behavior with dataSize > 4 and odd") {
      REQUIRE(CheckError(hipMemRangeGetAttribute(OutData, 7,
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, MEM_SIZE), __LINE__));
    }
    // checking the behavior with dataSize which is not multiple of 4
    SECTION("checking the behavior with dataSize which is not multiple of 4") {
      REQUIRE(CheckError(hipMemRangeGetAttribute(OutData, 27,
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, MEM_SIZE), __LINE__));
    }
    // checking the behaviour with devPtr(4th param) as NULL
    SECTION("checking the behaviour with devPtr(4th param) as NULL") {
    REQUIRE(CheckError(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeReadMostly,
                                     NULL, MEM_SIZE), __LINE__));
    }
    // checking the behaviour with count(5th param) as 0
    SECTION("checking the behaviour with count(5th param) as 0") {
      REQUIRE(CheckError(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeReadMostly,
                                     devPtr, 0), __LINE__));
    }
    // checking the behavior with invalid attribute (3rd param) as 0
    // as it is attribute hence avoiding the negative tests with 3rd param

    // checking the behaviour of the api with ptr allocated using
    // hipHostMalloc
    void *ptr = nullptr;
    SECTION("Checking behavior with hipHostMalloc ptr") {
      HIP_CHECK(hipHostMalloc(&ptr, MEM_SIZE, 0));
      REQUIRE(CheckError(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeReadMostly,
                                     ptr, MEM_SIZE), __LINE__));
      HIP_CHECK(hipHostFree(ptr));
    }
    HIP_CHECK(hipFree(devPtr));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/* This test case checks the behavior of hipMemRangeGetAttribute()  with
   AccessedBy flag is consistent with cuda's counter part*/
TEST_CASE("Unit_hipMemRangeGetAttribute_AccessedBy1") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int Ngpus = 0, *Hmm = NULL, MEM_SZ = 4096, RND_NUM = 999;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    int *OutData = new int[Ngpus];
    for (int i = 0; i < Ngpus; ++i) {
      OutData[Ngpus] = RND_NUM;
    }
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SZ));
    HIP_CHECK(hipMemAdvise(Hmm, MEM_SZ, hipMemAdviseSetAccessedBy, 0));
    HIP_CHECK(hipMemRangeGetAttribute(OutData, 4*Ngpus,
                                     hipMemRangeAttributeAccessedBy,
                                     Hmm, MEM_SZ));
    if (OutData[0] != 0) {
      WARN("Didn't receive expected value at line: " << __LINE__);
      REQUIRE(false);
    }
    for (int i = 1; i < Ngpus; ++i) {
      if (OutData[i] != -2) {
        WARN("Didn't receive expected value at line: " << __LINE__);
        REQUIRE(false);
      }
    }
    if (Ngpus >= 2) {
      for (int i = 0; i < Ngpus; ++i) {
        HIP_CHECK(hipMemAdvise(Hmm, MEM_SZ, hipMemAdviseSetAccessedBy, i));
      }
      // checking the behavior with dataSize less than the number of gpus
      // This should not result in segfault.
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4*(Ngpus-1),
                                       hipMemRangeAttributeAccessedBy,
                                       Hmm, MEM_SZ));
      // OutData should have stored the gpu ordinals for which AccessedBy is
      // assigned except for the last element which should have -2 stored
      // so as to be consistent with cuda's behavior
      for (int i = 0; i < (Ngpus - 1); ++i) {
        if (OutData[i] != i) {
          WARN("Didn't receive expected value at line: " << __LINE__);
          REQUIRE(false);
        }
      }
      if (OutData[Ngpus - 1] != -2) {
        WARN("Didn't receive expected value at line: " << __LINE__);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipFree(Hmm));
    delete[] OutData;
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}




/* Allocate  4 * page size of memory with the flag hipMemAttachGloal. Advise
   AccessedBy, ReadMostly and PreferredLocation to first half(2*pageSz) of the
    memory and probe the for the flags which are set earlier using
   hipMemRangeGetAttribute() api for the full size(4*PageSz).*/
/* Need to discuss the difference in behavior w.r.t cuda*/

TEST_CASE("Unit_hipMemRangeGetAttribte_3") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int Ngpus = 0, *Hmm = NULL, MEM_SZ = 4096*4, RND_NUM = 999;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    int *OutData = new int[Ngpus];
    for (int i = 0; i < Ngpus; ++i) {
      OutData[Ngpus] = RND_NUM;
    }
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SZ));
    HIP_CHECK(hipMemAdvise(Hmm, MEM_SZ/2, hipMemAdviseSetAccessedBy, 0));
    HIP_CHECK(hipMemRangeGetAttribute(OutData, 4*Ngpus,
                                     hipMemRangeAttributeAccessedBy,
                                     (Hmm), MEM_SZ));

    HIP_CHECK(hipMemAdvise(Hmm, MEM_SZ/2, hipMemAdviseSetReadMostly, 0));
    // The Api called below should not fail
    HIP_CHECK(hipMemRangeGetAttribute(OutData, 4,
                                     hipMemRangeAttributeReadMostly,
                                     (Hmm), MEM_SZ));

    HIP_CHECK(hipMemAdvise(Hmm, MEM_SZ/2, hipMemAdviseSetPreferredLocation, 0));
    // The api called below should not fail
    HIP_CHECK(hipMemRangeGetAttribute(OutData, 4,
                                     hipMemRangeAttributePreferredLocation,
                                     (Hmm), MEM_SZ));
    HIP_CHECK(hipFree(Hmm));
    delete[] OutData;
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* The following scenarios tests that probing the attributes which are not set
   by hipMemAdvise() but being probed using hipMemRangeGetAttribute() should
   not result in a crash*/

TEST_CASE("Unit_hipMemRangeGetAttribute_4") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, PageSz = 4096, Ngpus, RND_NUM = 999;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    int *OutData = new int[Ngpus];
    for (int i = 0; i < Ngpus; ++i) {
      OutData[i] = RND_NUM;
    }
    HIP_CHECK(hipMallocManaged(&Hmm, 4*PageSz));
    SECTION("Set ReadMostly & probe other flags") {
      HIP_CHECK(hipMemAdvise(Hmm, 4*PageSz, hipMemAdviseSetReadMostly, 0));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4*Ngpus,
                                       hipMemRangeAttributeAccessedBy,
                                       Hmm, 4*PageSz));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4,
                                       hipMemRangeAttributePreferredLocation,
                                       Hmm, 4*PageSz));
      HIP_CHECK(hipMemAdvise(Hmm, 4*PageSz, hipMemAdviseUnsetReadMostly, 0));
    }
    SECTION("Set AccessedBy & probe other flags") {
      HIP_CHECK(hipMemAdvise(Hmm, 4*PageSz, hipMemAdviseSetAccessedBy, 0));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4,
                                       hipMemRangeAttributeReadMostly,
                                       Hmm, 4*PageSz));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4,
                                       hipMemRangeAttributePreferredLocation,
                                       Hmm, 4*PageSz));
      HIP_CHECK(hipMemAdvise(Hmm, 4*PageSz, hipMemAdviseUnsetAccessedBy, 0));
    }
    SECTION("Set AccessedBy & probe other flags") {
      HIP_CHECK(hipMemAdvise(Hmm, 4*PageSz, hipMemAdviseSetPreferredLocation,
                            0));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4,
                                       hipMemRangeAttributeReadMostly,
                                       Hmm, 4*PageSz));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4*Ngpus,
                                       hipMemRangeAttributeAccessedBy,
                                       Hmm, 4*PageSz));
      HIP_CHECK(hipMemAdvise(Hmm, 4*PageSz, hipMemAdviseUnsetPreferredLocation,
                            0));
    }
    HIP_CHECK(hipFree(Hmm));
    delete[] OutData;
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* The following scenario is a simple test which does the following:
   Allocate Hmm memory --> hipMemPrefetchAsync() to device 0 and then
   probe LastPrefetchLocation attribute using hipMemRangeGetAttribute*/

TEST_CASE("Unit_hipMemRangeGetAttribute_PrefetchAndGtAttr") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int Ngpus = 0, *Hmm = NULL, RND_NUM = 999;
    size_t PageSz = 4096;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));

    int *OutData = new int[Ngpus];
    for (int i = 0; i < Ngpus; ++i) {
      OutData[Ngpus] = RND_NUM;
    }
    HIP_CHECK(hipMallocManaged(&Hmm, PageSz*4));
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipMemPrefetchAsync(Hmm, PageSz*4, 0, strm));
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipMemRangeGetAttribute(OutData, 4,
                                     hipMemRangeAttributeLastPrefetchLocation,
                                     Hmm, PageSz*4));
    HIP_CHECK(hipStreamDestroy(strm));
    HIP_CHECK(hipFree(Hmm));
    if (OutData[0] != 0) {
      WARN("Didnt receive expected value at line: " << __LINE__);
      delete[] OutData;
      REQUIRE(false);
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

