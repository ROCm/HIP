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

/* Test Case Description:
   Scenario 1: The  test validates if fine grain
   behavior is observed or not with memory allocated using hipHostMalloc()
   Scenario 2: The test validates if fine grain
   behavior is observed or not with memory allocated using hipMallocManaged()
   Scenario 3: The test validates if memory access is fine
   with memory allocated using hipMallocManaged() and CoarseGrain Advise
   Scenario 4: The test validates if memory access is fine
   with memory allocated using hipMalloc() and CoarseGrain Advise
   Scenario 5: The test validates if fine grain
   behavior is observed or not with memory allocated using
   hipExtMallocWithFlags()*/


#include <hip_test_common.hh>
#include <chrono>

__global__ void CoherentTst(int* ptr) {  // ptr was set to 1
  atomicAdd(ptr, 1);                     // now ptr is 2
  while (atomicCAS(ptr, 3, 4) != 3) {    // wait till ptr is 3, then change it to 4
  }
}

__global__  void SquareKrnl(int *ptr) {
  // ptr value squared here
  *ptr = (*ptr) * (*ptr);
}



// The variable below will work as signal to decide pass/fail
static bool YES_COHERENT = false;

// The function tests the coherency of allocated memory
// If this test hangs, means there is issue in coherency
static void TstCoherency(int* ptr, bool hmmMem) {
  int* dptr = nullptr;
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  // storing value 1 in the memory created above
  *ptr = 1;

  if (!hmmMem) {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dptr), ptr, 0));
    CoherentTst<<<1, 1, 0, stream>>>(dptr);
  } else {
    CoherentTst<<<1, 1, 0, stream>>>(ptr);
  }

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start)
                 .count() < 3 ||
         *ptr == 2) {
  }          // wait till ptr is 2 from kernel
  *ptr += 1; // increment it to 3

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  if (*ptr == 4) {
    YES_COHERENT = true;
  }
}

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


/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using hipHostMalloc()*/
// The following tests are disabled for Nvidia as they are not consistently
// passing
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_CoherentTst") {
  int *Ptr = nullptr,  SIZE = sizeof(int);
  bool HmmMem = false;
  YES_COHERENT = false;
  // Allocating hipHostMalloc() memory with hipHostMallocCoherent flag
  SECTION("hipHostMalloc with hipHostMallocCoherent flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocCoherent));
  }
  SECTION("hipHostMalloc with Default flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE));
  }
  SECTION("hipHostMalloc with hipHostMallocMapped flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocMapped));
  }

  TstCoherency(Ptr, HmmMem);
  HIP_CHECK(hipHostFree(Ptr));
  REQUIRE(YES_COHERENT);
}
#endif


/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using hipMallocManaged()*/
// The following tests are disabled for Nvidia as they are not consistently
// passing
#if HT_AMD
TEST_CASE("Unit_hipMallocManaged_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int);
  bool HmmMem = true;
  YES_COHERENT = false;

  int managed =  HmmAttrPrint();
  if (managed == 1) {
    // Allocating hipMallocManaged() memory
    SECTION("hipMallocManaged with hipMemAttachGlobal flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachGlobal));
    }
    SECTION("hipMallocManaged with hipMemAttachHost flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachHost));
    }
    TstCoherency(Ptr, HmmMem);
    HIP_CHECK(hipFree(Ptr));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
#endif

/* Test case description: The following test validates if memory access is fine
   with memory allocated using hipMallocManaged() and CoarseGrain Advise*/
TEST_CASE("Unit_hipMallocManaged_CoherentTstWthAdvise") {
  int *Ptr = nullptr, SIZE = sizeof(int);
  YES_COHERENT = false;
  // Allocating hipMallocManaged() memory
  SECTION("hipMallocManaged with hipMemAttachGlobal flag") {
    HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachGlobal));
  }
  SECTION("hipMallocManaged with hipMemAttachHost flag") {
    HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachHost));
  }
#if HT_AMD
  HIP_CHECK(hipMemAdvise(Ptr, SIZE, hipMemAdviseSetCoarseGrain, 0));
#endif
  // Initializing Ptr memory with 9
  *Ptr = 9;
  hipStream_t strm;
  HIP_CHECK(hipStreamCreate(&strm));
  SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
  HIP_CHECK(hipStreamSynchronize(strm));
  if (*Ptr == 81) {
    YES_COHERENT = true;
  }
  HIP_CHECK(hipFree(Ptr));
  HIP_CHECK(hipStreamDestroy(strm));
  REQUIRE(YES_COHERENT);
}


/* Test case description: The following test validates if memory allocated
   using hipMalloc() are of type Coarse Grain*/
// The following tests are disabled for Nvidia as they are not applicable
#if HT_AMD
TEST_CASE("Unit_hipMalloc_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int);
  uint32_t svm_attrib = 0;
  bool IfTstPassed = false;
  // Allocating hipMalloc() memory
  HIP_CHECK(hipMalloc(&Ptr, SIZE));
  HIP_CHECK(hipMemRangeGetAttribute(&svm_attrib, sizeof(svm_attrib),
        hipMemRangeAttributeCoherencyMode, Ptr, SIZE));
  if (svm_attrib == hipMemRangeCoherencyModeCoarseGrain) {
    IfTstPassed = true;
  }
  HIP_CHECK(hipFree(Ptr));
  REQUIRE(IfTstPassed);
}
#endif
/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using
   hipExtMallocWithFlags()*/
#if HT_AMD
TEST_CASE("Unit_hipExtMallocWithFlags_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int), InitVal = 9;
  bool FineGrain = true;
  YES_COHERENT = false;

  int managed =  HmmAttrPrint();
  if (managed == 1) {
    // Allocating hipExtMallocWithFlags() memory with flags
    SECTION("hipExtMallocWithFlags with hipDeviceMallocFinegrained flag") {
      HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE*2,
                                      hipDeviceMallocFinegrained));
    }
    SECTION("hipExtMallocWithFlags with hipDeviceMallocSignalMemory flag") {
      // for hipMallocSignalMemory flag the size of memory must be 8
      HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE*2,
                                      hipMallocSignalMemory));
    }
    SECTION("hipExtMallocWithFlags with hipDeviceMallocDefault flag") {
      /* hipExtMallocWithFlags() with flag
      hipDeviceMallocDefault allocates CoarseGrain memory */
      FineGrain = false;
      HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE*2,
                                      hipDeviceMallocDefault));
    }
    if (FineGrain) {
      TstCoherency(Ptr, FineGrain);
    } else {
      *Ptr = InitVal;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
      HIP_CHECK(hipStreamSynchronize(strm));
      if (*Ptr == (InitVal * InitVal)) {
        YES_COHERENT = true;
      }
    }
    HIP_CHECK(hipFree(Ptr));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
#endif

