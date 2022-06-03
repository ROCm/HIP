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

__global__  void CoherentTst(int *ptr, int PeakClk) {
  // Incrementing the value by 1
  int64_t GpuFrq = (PeakClk * 1000);
  int64_t StrtTck = clock64();
  atomicAdd(ptr, 1);
  // The following while loop checks the value in ptr for around 3-4 seconds
  while ((clock64() - StrtTck) <= (3 * GpuFrq)) {
    if (*ptr == 3) {
      atomicAdd(ptr, 1);
      return;
    }
  }
}

__global__  void SquareKrnl(int *ptr) {
  // ptr value squared here
  *ptr = (*ptr) * (*ptr);
}



// The variable below will work as signal to decide pass/fail
static bool YES_COHERENT = false;

// The function tests the coherency of allocated memory
static void TstCoherency(int *Ptr, bool HmmMem) {
  int *Dptr = nullptr, peak_clk;
  hipStream_t strm;
  HIP_CHECK(hipStreamCreate(&strm));
  // storing value 1 in the memory created above
  *Ptr = 1;
  // Getting gpu frequency
  HIP_CHECK(hipDeviceGetAttribute(&peak_clk, hipDeviceAttributeClockRate, 0));
  if (!HmmMem) {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&Dptr), Ptr,
                                      0));
    CoherentTst<<<1, 1, 0, strm>>>(Dptr, peak_clk);
  } else {
    CoherentTst<<<1, 1, 0, strm>>>(Ptr, peak_clk);
  }
  // looping until the value is 2 for 3 seconds
  std::chrono::steady_clock::time_point start =
               std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::seconds>(
         std::chrono::steady_clock::now() - start).count() < 3) {
    if (*Ptr == 2) {
      *Ptr += 1;
      break;
    }
  }
  HIP_CHECK(hipStreamSynchronize(strm));
  HIP_CHECK(hipStreamDestroy(strm));
  if (*Ptr == 4) {
    YES_COHERENT = true;
  }
}

/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using hipHostMalloc()*/
// The following tests are disabled for Nvidia as they are not consistently
// passing
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_CoherentTst") {
  int *Ptr = nullptr,  SIZE = sizeof(int), Pageable = 0;
  bool HmmMem = false;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  if (Pageable == 1) {
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
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributePageableMemoryAccess "
             "attribute. Hence skipping the test with Pass result.\n");
  }
}
#endif


/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using hipMallocManaged()*/
// The following tests are disabled for Nvidia as they are not consistently
// passing
#if HT_AMD
TEST_CASE("Unit_hipMallocManaged_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int), Pageable = 0, managed = 0;
  bool HmmMem = true;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);

  if (managed == 1 && Pageable == 1) {
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
    SUCCEED("GPU 0 doesn't support ManagedMemory or PageableMemoryAccess"
           "device attribute. Hence skipping the test with Pass result.\n");
  }
}
#endif

/* Test case description: The following test validates if memory access is fine
   with memory allocated using hipMallocManaged() and CoarseGrain Advise*/
TEST_CASE("Unit_hipMallocManaged_CoherentTstWthAdvise") {
  int *Ptr = nullptr, SIZE = sizeof(int), managed = 0;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);

  if (managed == 1) {
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
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the test with Pass result.\n");
  }
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
  int *Ptr = nullptr, SIZE = sizeof(int), InitVal = 9, Pageable = 0, managed = 0;
  bool FineGrain = true;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if (managed == 1 && Pageable == 1) {
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
    SUCCEED("GPU 0 doesn't support ManagedMemory or PageableMemoryAccess"
           "device attribute. Hence skipping the test with Pass result.\n");
  }
}
#endif

