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

/* Test Case Description:
   Scenario-1: The following Function Tests the working of flags which can be
   assigned to HMM memory using hipMemAdvise() api
   Scenario-2: Negative tests on hipMemAdvise() api
   Scenario-3: The following function tests various scenarios around the flag
   'hipMemAdviseSetPreferredLocation' using HMM memory and hipMemAdvise() api
   Scenario-4: The following function tests various scenarios around the flag
   'hipMemAdviseSetReadMostly' using HMM memory and hipMemAdvise() api
   Scenario-5: The following function verifies if assigning of a flag
   invalidates the earlier flag which was assigned to the same memory region
   using hipMemAdvise()
   Scenario-6: The following function tests if peers can set
   hipMemAdviseSetAccessedBy flag
   on HMM memory prefetched on each of the other gpus
   Scenario-7: Set AccessedBy flag and check value returned by
   hipMemRangeGetAttribute() It should be -2(same is observed on cuda)
   Scenario-8: Set AccessedBy flag to device 0 on Hmm memory and prefetch the
   memory to device 1, then probe for AccessedBy flag using
   hipMemRangeGetAttribute() we should still see the said flag is set for
   device 0
   Scenario-9: 1) Set AccessedBy to device 0 followed by PreferredLocation to
   device 1 check for AccessedBy flag using hipMemRangeGetAttribute() it should
   return 0
   2) Unset AccessedBy to 0 and set it to device 1 followed by
   PreferredLocation to device 1, check for AccessedBy flag using
   hipMemRangeGetAttribute() it should return 1
   Scenario-10: Set AccessedBy flag to HMM memory launch a kernel and then unset
   AccessedBy, launch kernel. We should not have any access issues
   Scenario-11: Allocate memory using aligned_alloc(), assign PreferredLocation
   flag to the allocated memory and launch a kernel. Kernel should get executed
   successfully without hang or segfault
   Scenario-12: Allocate Hmm memory, set advise to PreferredLocation and then
   get attribute using the api hipMemRangeGetAttribute() for
   hipMemRangeAttributeLastPrefetchLocation the value returned should be -2
   Scenario-13: Allocate HMM memory, set PreferredLocation to device 0, Prfetch
   the mem to device1, probe for hipMemRangeAttributeLastPrefetchLocation using
   hipMemRangeGetAttribute(), we should get 1
   Scenario-14: Allocate HMM memory, set ReadMostly followed by
   PreferredLocation, probe for hipMemRangeAttributeReadMostly and
   hipMemRangeAttributePreferredLocation
   using hipMemRangeGetAttribute() we should observe 1 and 0 correspondingly.
   In other words setting of hipMemRangeAttributePreferredLocation should not
   impact hipMemRangeAttributeReadMostly advise to the memory
   Scenario-15: Allocate Hmm memory, advise it to ReadMostly for gpu: 0 and
   launch kernel on all other gpus except 0. This test case may discover any
   effect or access denial case arising due to setting ReadMostly only to a
   particular gpu
*/

#include <hip_test_common.hh>
#if __linux__
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#endif

// Kernel function
__global__ void MemAdvseKernel(int n, int *x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    x[index] = x[index] * x[index];
}

// Kernel
__global__ void MemAdvise2(int *Hmm, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    Hmm[i] = Hmm[i] + 10;
  }
}

// Kernel
__global__ void MemAdvise3(int *Hmm, int *Hmm1, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    Hmm1[i] = Hmm[i] + 10;
  }
}


static bool CheckError(hipError_t err, int LineNo) {
  if (err == hipSuccess) {
    WARN("Error expected but received hipSuccess at line no.:" << LineNo);
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
  WARN("hipDeviceAttributePageableMemoryAccessUsesHostPageTables:" << managed);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  WARN("hipDeviceAttributeManagedMemory: " << managed);
  return managed;
}


// The following Function Tests the working of flags which can be assigned
// to HMM memory using hipMemAdvise() api
TEST_CASE("Unit_hipMemAdvise_TstFlags") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int NumDevs = 0, *Outpt = nullptr;
    int MEM_SIZE = 4*1024, A_CONST = 9999;
    float *Hmm = nullptr;
    int AttrVal = 0;
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    Outpt = new int[NumDevs];
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE * 2, hipMemAttachGlobal));
    // With the following for loop we iterate through each of the Gpus in the
    // system set and unset the flags and check the behavior.
    for (int i = 0; i < NumDevs; ++i) {
      HIP_CHECK(hipMemAdvise(Hmm , MEM_SIZE * 2, hipMemAdviseSetReadMostly, i));
      HIP_CHECK(hipMemRangeGetAttribute(&AttrVal, sizeof(AttrVal),
                                        hipMemRangeAttributeReadMostly, Hmm,
                                       MEM_SIZE * 2));
      if (AttrVal != 1) {
        WARN("Attempt to set hipMemAdviseSetReadMostly flag failed!\n");
        IfTestPassed = false;
      }
      HIP_CHECK(hipMemAdvise(Hmm , MEM_SIZE * 2, hipMemAdviseUnsetReadMostly,
                             i));

      HIP_CHECK(hipMemRangeGetAttribute(&AttrVal, sizeof(AttrVal),
                                       hipMemRangeAttributeReadMostly, Hmm,
                                       (MEM_SIZE * 2)));
      if (AttrVal != 0) {
        WARN("Attempt to Unset hipMemAdviseSetReadMostly flag failed!\n");
        IfTestPassed = false;
      }
      AttrVal = A_CONST;
      // Currently hipMemAdviseSetPreferredLocation and
      // hipMemAdviseSetAccessedBy
      // flags are resulting in issues: SWDEV-267357
      HIP_CHECK(hipMemAdvise(Hmm , MEM_SIZE * 2,
                            hipMemAdviseSetPreferredLocation, i));
      HIP_CHECK(hipMemRangeGetAttribute(&AttrVal, sizeof(AttrVal),
                                       hipMemRangeAttributePreferredLocation,
                                       Hmm, (MEM_SIZE * 2)));
      if (AttrVal != i) {
        WARN("Attempt to set hipMemAdviseSetPreferredLocation flag failed!\n");
        IfTestPassed = false;
      }
      AttrVal = A_CONST;
      HIP_CHECK(hipMemAdvise(Hmm , MEM_SIZE * 2,
                            hipMemAdviseUnsetPreferredLocation, i));
      HIP_CHECK(hipMemRangeGetAttribute(&AttrVal, sizeof(AttrVal),
                                       hipMemRangeAttributePreferredLocation,
                                       Hmm, (MEM_SIZE * 2)));
      if (AttrVal == i) {
      WARN("Attempt to Unset hipMemAdviseUnsetPreferredLocation ");
      WARN("flag failed!\n");
      IfTestPassed = false;
      }
      for (int m = 0; m < NumDevs; ++m) {
        Outpt[m] = A_CONST;
      }
      HIP_CHECK(hipMemAdvise(Hmm , MEM_SIZE * 2, hipMemAdviseSetAccessedBy, i));
      HIP_CHECK(hipMemRangeGetAttribute(Outpt, sizeof(Outpt),
                                       hipMemRangeAttributeAccessedBy, Hmm,
                                       (MEM_SIZE * 2)));
      if ((Outpt[0]) != i) {
        WARN("Attempt to set hipMemAdviseSetAccessedBy flag failed!\n");
        IfTestPassed = false;
      }
      for (int m = 0; m < NumDevs; ++m) {
        Outpt[m] = A_CONST;
      }
      HIP_CHECK(hipMemAdvise(Hmm , MEM_SIZE * 2, hipMemAdviseUnsetAccessedBy,
                             i));
      HIP_CHECK(hipMemRangeGetAttribute(Outpt, sizeof(Outpt),
                                       hipMemRangeAttributeAccessedBy, Hmm,
                                       (MEM_SIZE * 2)));
      if ((Outpt[0]) >= 0) {
        WARN("Attempt to Unset hipMemAdviseUnsetAccessedBy flag failed!\n");
        IfTestPassed = false;
      }
    }
    delete [] Outpt;
    HIP_CHECK(hipFree(Hmm));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

TEST_CASE("Unit_hipMemAdvise_NegtveTsts") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int NumDevs = 0, MEM_SIZE = 4*1024;
    float *Hmm = nullptr;
    std::string str;
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE * 2, hipMemAttachGlobal));
#if HT_AMD
    // Passing invalid value(99) device param
    IfTestPassed &= CheckError(hipMemAdvise(Hmm, MEM_SIZE * 2,
                               hipMemAdviseSetReadMostly, 99), __LINE__);

    // Passing invalid value(-12) device param
    IfTestPassed &= CheckError(hipMemAdvise(Hmm, MEM_SIZE * 2,
                               hipMemAdviseSetReadMostly, -12), __LINE__);
#endif
    // Passing NULL as first parameter instead of valid pointer to a memory
    IfTestPassed &= CheckError(hipMemAdvise(NULL, MEM_SIZE * 2,
                               hipMemAdviseSetReadMostly, 0), __LINE__);

    // Passing 0 for count(2nd param) parameter
    IfTestPassed &= CheckError(hipMemAdvise(Hmm, 0, hipMemAdviseSetReadMostly,
                                            0), __LINE__);

    // Passing count much more than actually allocated value
    IfTestPassed &= CheckError(hipMemAdvise(Hmm, MEM_SIZE * 6,
                               hipMemAdviseSetReadMostly, 0), __LINE__);

    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

// The following function tests various scenarios around the flag
// 'hipMemAdviseSetPreferredLocation' using HMM memory and hipMemAdvise() api
TEST_CASE("Unit_hipMemAdvise_PrefrdLoc") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    // Check that when a page fault occurs for the memory region set to devPtr,
    // the data is migrated to the destn processor
    int MEM_SIZE = 4096, A_CONST = 9999;
    int *Hmm = nullptr, NumDevs = 0, dev = A_CONST;
    bool IfTestPassed = true;
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE * 3, hipMemAttachGlobal));
    for (int i = 0; i < ((MEM_SIZE * 3)/4); ++i) {
        Hmm[i]  = 4;
    }
    for (int devId = 0; devId < NumDevs; ++devId) {
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE * 3,
                             hipMemAdviseSetPreferredLocation, devId));
      int NumElms = ((MEM_SIZE * 3)/4);
      MemAdvseKernel<<<NumElms/32, 32>>>(NumElms, Hmm);
      int dev = A_CONST;
      HIP_CHECK(hipMemRangeGetAttribute(&dev, sizeof(dev),
                                       hipMemRangeAttributePreferredLocation,
                                       Hmm, MEM_SIZE * 3));
      if (dev != devId) {
        WARN("Memory observed to be not available on expected location\n");
        WARN("line no: " << __LINE__);
        WARN("dev: " << dev);
        IfTestPassed = false;
      }
    }

    // Check that when preferred location is set for a memory region,
    // data can still be prefetched using hipMemPrefetchAsync
    hipStream_t strm;
    dev = A_CONST;
    for (int devId = 0; devId < NumDevs; ++devId) {
      HIP_CHECK(hipSetDevice(devId));
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE * 3,
                             hipMemAdviseSetPreferredLocation, devId));
      HIP_CHECK(hipMemPrefetchAsync(Hmm, MEM_SIZE * 3, devId, strm));
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipMemRangeGetAttribute(&dev, sizeof(dev),
                                       hipMemRangeAttributeLastPrefetchLocation,
                                       Hmm, MEM_SIZE * 3));
      if (dev != devId) {
        WARN("Memory reported to be not available at the Prefetched ");
        WARN("location with device id: " << devId);
        WARN("line no: " << __LINE__);
        WARN("dev: " << dev);
        IfTestPassed = false;
      }
      HIP_CHECK(hipStreamDestroy(strm));
    }
    HIP_CHECK(hipFree(Hmm));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

// The following function tests various scenarios around the flag
// 'hipMemAdviseSetReadMostly' using HMM memory and hipMemAdvise() api

TEST_CASE("Unit_hipMemAdvise_ReadMostly") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int MEM_SIZE = 4096, A_CONST = 9999;
    float *Hmm = nullptr;
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE));
    for (uint64_t i = 0; i < (MEM_SIZE/sizeof(float)); ++i) {
      Hmm[i] = A_CONST;
    }
    HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetReadMostly, 0));
    // Checking if the data can be read after setting hipMemAdviseSetReadMostly
    for (uint64_t i = 0; i < (MEM_SIZE/sizeof(float)); ++i) {
      if (Hmm[i] != A_CONST) {
        WARN("Didn't find expected value in Hmm memory after setting");
        WARN(" hipMemAdviseSetReadMostly flag line no.: " << __LINE__);
        IfTestPassed = false;
      }
    }

    // Checking if the memory region can be modified
    for (uint64_t i = 0; i < (MEM_SIZE/sizeof(float)); ++i) {
      Hmm[i] = A_CONST;
    }

    for (uint64_t i = 0; i < (MEM_SIZE/sizeof(float)); ++i) {
      if (Hmm[i] != A_CONST) {
        WARN("Didn't find expected value in Hmm memory after Modification\n");
        WARN("line no.: " << __LINE__);
        IfTestPassed = false;
      }
    }

    int out = A_CONST;
    HIP_CHECK(hipMemRangeGetAttribute(&out, 4, hipMemRangeAttributeReadMostly,
                                      Hmm, MEM_SIZE));
    if (out != 1) {
      WARN("out value: " << out);
      IfTestPassed = false;
    }
    // Checking the advise attribute after prefetch
    HIP_CHECK(hipMemPrefetchAsync(Hmm, MEM_SIZE, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemRangeGetAttribute(&out, sizeof(int),
                                     hipMemRangeAttributeReadMostly, Hmm,
                                     MEM_SIZE));
    if (out != 1) {
      WARN("Attribute assigned to memory changed after calling ");
      WARN("hipMemPrefetchAsync(). line no.: " << __LINE__);
      WARN("out value: " << out);
      IfTestPassed = false;
    }
    HIP_CHECK(hipFree(Hmm));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

// The following function verifies if assigning of a flag invalidates the
// earlier flag which was assigned to the same memory region using
// hipMemAdvise()
TEST_CASE("Unit_hipMemAdvise_TstFlgOverrideEffect") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int MEM_SIZE = 4*4096, A_CONST = 9999;
    float *Hmm = nullptr;
    int NumDevs = 0, dev = A_CONST;

    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE, hipMemAttachGlobal));
    for (int i = 0; i < NumDevs; ++i) {
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetReadMostly, i));
      HIP_CHECK(hipMemRangeGetAttribute(&dev, sizeof(int),
                                       hipMemRangeAttributeReadMostly, Hmm,
                                       MEM_SIZE));
      if (dev != 1) {
        WARN("hipMemAdviseSetReadMostly flag did not take affect despite ");
        WARN("setting it using hipMemAdvise(). line no.: " << __LINE__);
        IfTestPassed = false;
        break;
      }
      dev = A_CONST;
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetPreferredLocation,
                             i));
      HIP_CHECK(hipMemRangeGetAttribute(&dev, sizeof(int),
                                       hipMemRangeAttributePreferredLocation,
                                       Hmm, MEM_SIZE));
      if (dev != i) {
        WARN("hipMemAdviseSetPreferredLocation flag did not take affect ");
        WARN("despite setting it using hipMemAdvise()\n");
        WARN("line no.: " << __LINE__);
        IfTestPassed = false;
        break;
      }

      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetAccessedBy, i));
      dev = A_CONST;
      HIP_CHECK(hipMemRangeGetAttribute(&dev, sizeof(int),
                                       hipMemRangeAttributeAccessedBy, Hmm,
                                       MEM_SIZE));
      if (dev != i) {
        WARN("hipMemAdviseSetAccessedBy flag did not take affect despite ");
        WARN("setting it using hipMemAdvise(). line no.: " << __LINE__);
        IfTestPassed = false;
        break;
      }
      HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseUnsetAccessedBy, i));
    }
    HIP_CHECK(hipFree(Hmm));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


// The following function tests if peers can set hipMemAdviseSetAccessedBy flag
// on HMM memory prefetched on each of the other gpus
#if HT_AMD
TEST_CASE("Unit_hipMemAdvise_TstAccessedByPeer") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int *Hmm = nullptr, MEM_SIZE = 4*4096, A_CONST = 9999;;
    int NumDevs = 0, CanAccessPeer = A_CONST, flag = 0;

    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    if (NumDevs < 2) {
      SUCCEED("Test TestSetAccessedByPeer() need atleast two Gpus to test"
             " the scenario. This system has GPUs less than 2");
    }
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE, hipMemAttachGlobal));
    for (int i = 0; i < NumDevs; ++i) {
      HIP_CHECK(hipMemPrefetchAsync(Hmm, MEM_SIZE, i, 0));
      for (int j = 0; j < NumDevs; ++j) {
        if (i == j)
          continue;
        HIP_CHECK(hipSetDevice(j));
        HIP_CHECK(hipDeviceCanAccessPeer(&CanAccessPeer, j, i));
        if (CanAccessPeer) {
          HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetAccessedBy, j));
          for (uint64_t m = 0; m < (MEM_SIZE/sizeof(int)); ++m) {
            Hmm[m] = 4;
          }
          HIP_CHECK(hipDeviceEnablePeerAccess(i, 0));
          MemAdvseKernel<<<(MEM_SIZE/sizeof(int)/32), 32>>>(
                           (MEM_SIZE/sizeof(int)), Hmm);
          HIP_CHECK(hipDeviceSynchronize());
          // Verifying the result
          for (uint64_t m = 0; m < (MEM_SIZE/sizeof(int)); ++m) {
            if (Hmm[m] != 16) {
              flag = 1;
            }
          }
          if (flag) {
            WARN("Didnt get Expected results with device: " << j);
            WARN("line no.: " << __LINE__);
            IfTestPassed = false;
            flag = 0;
          }
        }
      }
    }
    HIP_CHECK(hipFree(Hmm));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
#endif


/* Set AccessedBy flag and check value returned by hipMemRangeGetAttribute()
   It should be -2(same is observed on cuda)*/
TEST_CASE("Unit_hipMemAdvise_TstAccessedByFlg") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999;
    HIP_CHECK(hipMallocManaged(&Hmm, 2*4096));
    HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseSetAccessedBy, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeLastPrefetchLocation,
                                     Hmm, 2*4096));
    if (data != -2) {
      WARN("Didnt get expected value!!\n");
      REQUIRE(false);
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/* Set AccessedBy flag to device 0 on Hmm memory and prefetch the memory to
   device 1, then probe for AccessedBy flag using hipMemRangeGetAttribute()
   we should still see the said flag is set for device 0*/
TEST_CASE("Unit_hipMemAdvise_TstAccessedByFlg2") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999, Ngpus = 0;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    if (Ngpus >= 2) {
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMallocManaged(&Hmm, 2*4096));
      HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseSetAccessedBy, 0));
      HIP_CHECK(hipMemPrefetchAsync(Hmm, 2*4096, 1, strm));
      HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                hipMemRangeAttributeAccessedBy, Hmm, 2*4096));
      if (data != 0) {
        WARN("Didnt get expected behavior at line: " << __LINE__);
        REQUIRE(false);
      }
      HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseUnsetAccessedBy, 0));
      HIP_CHECK(hipStreamDestroy(strm));
      HIP_CHECK(hipFree(Hmm));
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}



/* 1) Set AccessedBy to device 0 followed by PreferredLocation to device 1
   check for AccessedBy flag using hipMemRangeGetAttribute() it should
   return 0
   2) Unset AccessedBy to 0 and set it to device 1 followed by
   PreferredLocation to device 1, check for AccessedBy flag using
   hipMemRangeGetAttribute() it should return 1*/

TEST_CASE("Unit_hipMemAdvise_TstAccessedByFlg3") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999, Ngpus = 0;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    if (Ngpus >= 2) {
      HIP_CHECK(hipMallocManaged(&Hmm, 2*4096));
      HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseSetAccessedBy, 0));
      HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseSetPreferredLocation, 1));
      HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                hipMemRangeAttributeAccessedBy, Hmm, 2*4096));
      if (data != 0) {
        WARN("Didnt get expected behavior at line: " << __LINE__);
        REQUIRE(false);
      }
      HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseUnsetAccessedBy, 0));
      HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseSetAccessedBy, 1));
      HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseSetPreferredLocation, 0));
      HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                hipMemRangeAttributeAccessedBy, Hmm, 2*4096));
      if (data != 1) {
        WARN("Didnt get expected behavior at line: " << __LINE__);
        REQUIRE(false);
      }
      HIP_CHECK(hipFree(Hmm));
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* Set AccessedBy flag to HMM memory launch a kernel and then unset
   AccessedBy, launch kernel. We should not have any access issues*/

TEST_CASE("Unit_hipMemAdvise_TstAccessedByFlg4") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, NumElms = (1024 * 1024), InitVal = 123, blockSize = 64;
    int DataMismatch = 0;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipMallocManaged(&Hmm, (NumElms * sizeof(int))));
    HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)),
                           hipMemAdviseSetAccessedBy, 0));
    // Initializing memory
    for (int i = 0; i < NumElms; ++i) {
      Hmm[i] = InitVal;
    }
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
    // launching kernel from each one of the gpus
    MemAdvise2<<<dimGrid, dimBlock, 0, strm>>>(Hmm, NumElms);
    HIP_CHECK(hipStreamSynchronize(strm));

    // verifying the final result
    for (int i = 0; i < NumElms; ++i) {
      if (Hmm[i] != (InitVal + 10)) {
        DataMismatch++;
      }
    }

    if (DataMismatch != 0) {
      WARN("DataMismatch is observed at line: " << __LINE__);
      REQUIRE(false);
    }

    HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)),
                           hipMemAdviseUnsetAccessedBy, 0));
    MemAdvise2<<<dimGrid, dimBlock, 0, strm>>>(Hmm, NumElms);
    HIP_CHECK(hipStreamSynchronize(strm));
    // verifying the final result
    for (int i = 0; i < NumElms; ++i) {
      if (Hmm[i] != (InitVal + (2*10))) {
        DataMismatch++;
      }
    }

    if (DataMismatch != 0) {
      WARN("DataMismatch is observed at line: " << __LINE__);
      REQUIRE(false);
    }
    HIP_CHECK(hipFree(Hmm));
    HIP_CHECK(hipStreamDestroy(strm));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* Allocate memory using aligned_alloc(), assign PreferredLocation flag to
   the allocated memory and launch a kernel. Kernel should get executed
   successfully without hang or segfault*/
#if __linux__ && HT_AMD
TEST_CASE("Unit_hipMemAdvise_TstAlignedAllocMem") {
  if ((setenv("HSA_XNACK", "1", 1)) != 0) {
    WARN("Unable to turn on HSA_XNACK, hence terminating the Test case!");
    REQUIRE(false);
  }
  // The following code block checks for gfx90a so as to skip if the device is not MI200

  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);

  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    int stat = 0;
    if (fork() == 0) {
      // The below part should be inside fork
      int managedMem = 0, pageMemAccess = 0;
      HIP_CHECK(hipDeviceGetAttribute(&pageMemAccess,
                hipDeviceAttributePageableMemoryAccess, 0));
      WARN("hipDeviceAttributePageableMemoryAccess:" << pageMemAccess);

      HIP_CHECK(hipDeviceGetAttribute(&managedMem, hipDeviceAttributeManagedMemory, 0));
      WARN("hipDeviceAttributeManagedMemory: " << managedMem);
      if ((managedMem == 1) && (pageMemAccess == 1)) {
        int *Mllc = nullptr, MemSz = 4096 * 4, NumElms = 4096, InitVal = 123;
        // Mllc = reinterpret_cast<(int *)>(aligned_alloc(4096, MemSz));
        Mllc = reinterpret_cast<int*>(aligned_alloc(4096, 4096*4));
        for (int i = 0; i < NumElms; ++i) {
          Mllc[i] = InitVal;
        }
        hipStream_t strm;
        int DataMismatch = 0;
        HIP_CHECK(hipStreamCreate(&strm));
        // The following hipMemAdvise() call is made to know if advise on
        // aligned_alloc() is causing any issue
        HIP_CHECK(hipMemAdvise(Mllc, MemSz, hipMemAdviseSetPreferredLocation, 0));
        HIP_CHECK(hipMemPrefetchAsync(Mllc, MemSz, 0, strm));
        HIP_CHECK(hipStreamSynchronize(strm));
        MemAdvise2<<<(NumElms/32), 32, 0, strm>>>(Mllc, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        for (int i = 0; i < NumElms; ++i) {
          if (Mllc[i] != (InitVal + 10)) {
            DataMismatch++;
          }
        }
        if (DataMismatch != 0) {
          WARN("DataMismatch observed!!");
          exit(9);  // 9 for failure
        } else {
          exit(10);  // 10 for Pass result
        }
      } else {
        SUCCEED("GPU 0 doesn't support ManagedMemory with hipDeviceAttributePageableMemoryAccess "
                "attribute. Hence skipping the testing with Pass result.\n");
        exit(Catch::ResultDisposition::ContinueOnFailure);
      }
    } else {
      wait(&stat);
      int Result = WEXITSTATUS(stat);
      if (Result == Catch::ResultDisposition::ContinueOnFailure) {
        WARN("GPU 0 doesn't support ManagedMemory with hipDeviceAttributePageableMemoryAccess "
             "attribute. Hence skipping the testing with Pass result.\n");
      } else {
        if (Result != 10) {
          REQUIRE(false);
        }
      }
    }
  } else {
      SUCCEED("Memory model feature is only supported for gfx90a, Hence"
              "skipping the testcase for this GPU " << device);
      WARN("Memory model feature is only supported for gfx90a, Hence"
              "skipping the testcase for this GPU " << device);
  }

}
#endif

/* Allocate Hmm memory, set advise to PreferredLocation and then get
   attribute using the api hipMemRangeGetAttribute() for
   hipMemRangeAttributeLastPrefetchLocation the value returned should be -2*/

TEST_CASE("Unit_hipMemAdvise_TstMemAdvisePrefrdLoc") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999;
    HIP_CHECK(hipMallocManaged(&Hmm, 4096));
    HIP_CHECK(hipMemAdvise(Hmm, 4096, hipMemAdviseSetPreferredLocation, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeLastPrefetchLocation,
                                     Hmm, 4096));
    if (data != -2) {
      WARN("Didnt receive expected value.");
      REQUIRE(false);
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* Allocate HMM memory, set PreferredLocation to device 0, Prfetch the mem
   to device1, probe for hipMemRangeAttributeLastPrefetchLocation using
   hipMemRangeGetAttribute(), we should get 1*/

TEST_CASE("Unit_hipMemAdvise_TstMemAdviseLstPreftchLoc") {
  int NumDevs = 0;
  HIP_CHECK(hipGetDeviceCount(&NumDevs));
  if (NumDevs >= 2) {
    int managed = HmmAttrPrint();
    if (managed == 1) {
      int *Hmm = NULL, data = 999;
      hipStream_t strm;
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMallocManaged(&Hmm, 4096));
      HIP_CHECK(hipMemAdvise(Hmm, 4096, hipMemAdviseSetPreferredLocation, 0));
      HIP_CHECK(hipMemPrefetchAsync(Hmm, 4096, 1, strm));
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                                       hipMemRangeAttributeLastPrefetchLocation,
                                       Hmm, 4096));
      if (data != 1) {
        WARN("Didnt receive expected value!!");
        REQUIRE(false);
      }
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the testing with Pass result.\n");
    }
  } else {
    SUCCEED("This system has less than 2 gpus hence skipping the test.\n");
  }
}


/* Allocate HMM memory, set ReadMostly followed by PreferredLocation, probe
   for hipMemRangeAttributeReadMostly and hipMemRangeAttributePreferredLocation
   using hipMemRangeGetAttribute() we should observe 1 and 0 correspondingly.
   In other words setting of hipMemRangeAttributePreferredLocation should not
   impact hipMemRangeAttributeReadMostly advise to the memory*/

TEST_CASE("Unit_hipMemAdvise_TstMemAdviseMultiFlag") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999;
    HIP_CHECK(hipMallocManaged(&Hmm, 4096));
    HIP_CHECK(hipMemAdvise(Hmm, 4096, hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemAdvise(Hmm, 4096, hipMemAdviseSetPreferredLocation, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributeReadMostly, Hmm,
                                     4096));
    if (data != 1) {
      WARN("Didnt receive expected value at line: " << data);
      REQUIRE(false);
    }
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                                     hipMemRangeAttributePreferredLocation, Hmm,
                                     4096));
    if (data != 0) {
      WARN("Didnt receive expected value at line: " << data);
      REQUIRE(false);
    }
    HIP_CHECK(hipFree(Hmm));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}



/*Allocate Hmm memory, advise it to ReadMostly for gpu: 0 and launch kernel
  on all other gpus except 0. This test case may discover any effect or
  access denial case arising due to setting ReadMostly only to a particular
  gpu*/

TEST_CASE("Unit_hipMemAdvise_ReadMosltyMgpuTst") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int Ngpus = 0;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    if (Ngpus < 2) {
      SUCCEED("This test needs atleast two gpus to run."
      "Hence skipping the test.\n");
    }
    int *Hmm = NULL, NumElms = (1024 * 1024), InitVal = 123, blockSize = 64;
    int *Hmm1 = NULL, DataMismatch = 0;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipMallocManaged(&Hmm, (NumElms * sizeof(int))));
    // Initializing memory
    for (int i = 0; i < NumElms; ++i) {
      Hmm[i] = InitVal;
    }
    HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)),
                           hipMemAdviseSetReadMostly, 0));
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
#if HT_AMD
    SECTION("Launch Kernel on all other gpus") {
      // launching kernel from each one of the gpus
      for (int i = 1; i < Ngpus; ++i) {
        DataMismatch = 0;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMallocManaged(&Hmm1, (NumElms * sizeof(int))));
        MemAdvise3<<<dimGrid, dimBlock, 0, strm>>>(Hmm, Hmm1, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        // verifying the results
        for (int j = 0; j < NumElms; ++j) {
          if (Hmm1[j] != (InitVal + 10)) {
            DataMismatch++;
          }
        }
        if (DataMismatch != 0) {
          WARN("DataMismatch is observed with the gpu: " << i);
          REQUIRE(false);
        }
        HIP_CHECK(hipFree(Hmm1));
      }
    }

    SECTION("Launch Kernel on all other gpus and manipulate the content") {
      for (int i = 0; i < Ngpus; ++i) {
        DataMismatch = 0;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)),
                               hipMemAdviseSetReadMostly, i));
        MemAdvise2<<<dimGrid, dimBlock, 0, strm>>>(Hmm, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
      }
      // verifying the final result
      for (int i = 0; i < NumElms; ++i) {
        if (Hmm[i] != (InitVal + Ngpus * 10)) {
          DataMismatch++;
        }
      }

      if (DataMismatch != 0) {
        WARN("DataMismatch is observed at line: " << __LINE__);
        REQUIRE(false);
      }
    }
#endif
    HIP_CHECK(hipFree(Hmm));
    HIP_CHECK(hipStreamDestroy(strm));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


TEST_CASE("Unit_hipMemAdvise_TstSetUnsetPrfrdLoc") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999;
    HIP_CHECK(hipMallocManaged(&Hmm, 2*4096));
    HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseSetPreferredLocation, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                        hipMemRangeAttributePreferredLocation, Hmm, 2*4096));
    if (data != 0) {
      WARN("Didnt receive expected value!!");
      REQUIRE(false);
    }
    HIP_CHECK(hipMemAdvise(Hmm, 2*4096, hipMemAdviseUnsetPreferredLocation, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int),
                          hipMemRangeAttributePreferredLocation, Hmm, 2*4096));
    if (data != -2) {
      WARN("Didnt receive expected value!!");
      REQUIRE(false);
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

