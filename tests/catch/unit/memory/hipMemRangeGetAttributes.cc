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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* Test Case Description:
   Scenario-1: Testing basic working of hipMemRangeGetAttributes()
   api with different flags
   Scenario-2: Negative testing with hipMemRangeGetAttributes() api
*/

#include <hip_test_common.hh>
#define MEM_SIZE 8192

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

#ifdef __linux__
/* Test Scenario: Testing basic working of hipMemRangeGetAttributes()
   api with different flags */

TEST_CASE("Unit_hipMemRangeGetAttributes_TstFlgs") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int NumDevs = 0;
    int *Outpt[4], *AcsdBy = nullptr;
    float *Hmm = nullptr;
    hipStream_t strm;
    hipMemRangeAttribute AttrArr[4] =
                                {hipMemRangeAttributeReadMostly,
                                hipMemRangeAttributePreferredLocation,
                                hipMemRangeAttributeAccessedBy,
                                hipMemRangeAttributeLastPrefetchLocation};
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    AcsdBy = new int(NumDevs);
    size_t dataSizes[4] = {sizeof(int), sizeof(int),
                        (NumDevs * sizeof(int)), sizeof(int)};
    Outpt[0] = new int;
    Outpt[1] = new int;
    Outpt[2] = new int[NumDevs];
    Outpt[3] = new int;
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE, hipMemAttachGlobal));
    for (int i = 0; i < NumDevs; ++i) {
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetReadMostly, i));
      HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(Outpt),
                                        dataSizes, AttrArr, 4, Hmm,
                                        MEM_SIZE));
      if (*(Outpt[0]) != 1) {
        WARN("Attempt to set hipMemAdviseSetReadMostly flag failed!\n");
        IfTestPassed = false;
      }
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseUnsetReadMostly, i));
      HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(Outpt),
                                    reinterpret_cast<size_t*>(dataSizes),
                                    AttrArr, 4, Hmm, MEM_SIZE));

      if (*(Outpt[0]) != 0) {
        WARN("Attempt to set hipMemAdviseUnsetReadMostly flag failed!\n");
        IfTestPassed = false;
      }

      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE,
                            hipMemAdviseSetPreferredLocation, i));
      HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(Outpt),
                                    reinterpret_cast<size_t*>(dataSizes),
                                    AttrArr, 4, Hmm, MEM_SIZE));
      if (*(Outpt[1]) != i) {
        WARN("Attempt to set hipMemAdviseSetPreferredLocation flag");
        WARN(" failed!\n");
        IfTestPassed = false;
      }
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetAccessedBy, i));
      HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(Outpt),
                                     reinterpret_cast<size_t*>(dataSizes),
                                     AttrArr, 4, Hmm, MEM_SIZE));
      if ((Outpt[2][0]) != i) {
        WARN("Attempt to set hipMemAdviseSetAccessedBy flag");
        WARN(" failed!\n");
        IfTestPassed = false;
      }

      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseUnsetAccessedBy, i));
      HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(Outpt),
                                     reinterpret_cast<size_t*>(dataSizes),
                                     AttrArr, 4, Hmm, MEM_SIZE));
      if (!((Outpt[2][i]) < 0)) {
        WARN("Attempt to set hipMemAdviseUnsetAccessedBy flag failed!\n");
        IfTestPassed = false;
      }
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMemPrefetchAsync(Hmm, MEM_SIZE, i, strm));
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(Outpt),
                                     reinterpret_cast<size_t*>(dataSizes),
                                     AttrArr, 4, Hmm, MEM_SIZE));
      if (*(Outpt[3]) != i) {
        WARN("Attempt to prefetch memory to device: " << i);
        WARN("failed!\n");
        IfTestPassed = false;
      }
      // Prefetching back to Host
      HIP_CHECK(hipMemPrefetchAsync(Hmm, MEM_SIZE, -1, strm));
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(Outpt),
                                     reinterpret_cast<size_t*>(dataSizes),
                                     AttrArr, 4, Hmm, MEM_SIZE));
      if (*(Outpt[3]) != -1) {
        WARN("Attempt to prefetch memory to Host failed!\n");
        IfTestPassed = false;
      }
    }

    HIP_CHECK(hipFree(Hmm));
    delete[] AcsdBy;
    for (int i = 0; i < 4; ++i) {
      delete Outpt[i];
    }
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/* Test Scenario: Negative testing with hipMemRangeGetAttributes() api*/
TEST_CASE("Unit_hipMemRangeGetAttributes_NegativeTst") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int NumDevs = 0, *Outpt[4];
    float *Hmm = nullptr;
    hipMemRangeAttribute AttrArr[4] =
                                {hipMemRangeAttributeReadMostly,
                                hipMemRangeAttributePreferredLocation,
                                hipMemRangeAttributeAccessedBy,
                                hipMemRangeAttributeLastPrefetchLocation};
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    size_t dataSizes[4] = {sizeof(int), sizeof(int),
                        (NumDevs * sizeof(int)), sizeof(int)};
    Outpt[0] = new int;
    Outpt[1] = new int;
    Outpt[2] = new int[NumDevs];
    Outpt[3] = new int;
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE, hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(Hmm , MEM_SIZE, hipMemAdviseSetReadMostly, 0));
    // passing zero for num of attributes param(4th)
    SECTION("passing zero for num of attributes param(4th)") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr, 0, Hmm, MEM_SIZE), __LINE__)) {
        IfTestPassed = false;
      }
    }

    // the first dataSize element passed as 0
    dataSizes[0] = 0;
    dataSizes[1] = sizeof(int);
    dataSizes[2] = NumDevs * sizeof(int);
    dataSizes[3] = sizeof(int);
    SECTION("the first dataSize element passed as 0") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr, 4, Hmm, MEM_SIZE),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    // passing datasize as 2 while the requirement is multiple of 4
    dataSizes[0] = 2;
    dataSizes[1] = sizeof(int);
    dataSizes[2] = NumDevs * sizeof(int);
    dataSizes[3] = sizeof(int);
    SECTION("datasize as 2 while the requirement is multiple of 4") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr, 4, Hmm, MEM_SIZE),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    // passing datasize as 6 while the requirement is multiple of 4
    dataSizes[0] = 6;
    dataSizes[1] = sizeof(int);
    dataSizes[2] = NumDevs * sizeof(int);
    dataSizes[3] = sizeof(int);
    SECTION("datasize as 6 while the requirement is multiple of 4") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr, 4, Hmm, MEM_SIZE),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    // passing datasize as 7 while the requirement is multiple of 4
    dataSizes[0] = 7;
    dataSizes[1] = sizeof(int);
    dataSizes[2] = NumDevs * sizeof(int);
    dataSizes[3] = sizeof(int);
    SECTION("datasize as 7 while the requirement is multiple of 4") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr, 4, Hmm, MEM_SIZE),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    // passing dataSize as 7 for attribute hipMemRangeAttributeAccessedBy
    hipMemRangeAttribute AttrArr1[1] = {hipMemRangeAttributeAccessedBy};
    dataSizes[2] = {7};
    SECTION("passing dataSize as 7 for attribute hipMemRangeAttrAccessedBy") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr1, 1, Hmm, MEM_SIZE), __LINE__)) {
        IfTestPassed = false;
      }
    }
    // Passing NULL as first parameter
    SECTION("Passing NULL as first parameter") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(NULL),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr, 4, Hmm, MEM_SIZE),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    // Passing count parameter as zero
    SECTION("Passing count parameter as zero") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      AttrArr, 4, Hmm, 0),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    // Passing NULL for Attribute array(3rd param)
    SECTION("Passing NULL for Attribute array(3rd param)") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      NULL, 4, Hmm, MEM_SIZE),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    // Passing 0 for Attribute array(3rd param)
    SECTION("Passing 0 for Attribute array(3rd param)") {
      if (!CheckError(hipMemRangeGetAttributes(
                                      reinterpret_cast<void**>(Outpt),
                                      reinterpret_cast<size_t*>(dataSizes),
                                      0, 4, Hmm, MEM_SIZE),
                                      __LINE__)) {
        IfTestPassed = false;
      }
    }
    for (int i = 0; i < 4; ++i) {
      delete Outpt[i];
    }
    REQUIRE(IfTestPassed);

    // The following scenarios have been removed considering the nature of the
    //  api. With Consultation with Maneesh Gupta, the following scenarios
    //   have been removed.
    // passing numAttributes as 4 while the attributes array has only 2 members
    // passing numAttributes as 10 while the attributes array has only 2 members
    // length of the list of dataSizes less than the number of
    // attributes being probed
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
#endif
