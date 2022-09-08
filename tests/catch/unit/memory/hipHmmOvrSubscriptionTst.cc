/*
Copyright (c) 2021-Present Advanced Micro Devices, Inc. All rights reserved.

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

/* Test Case Description: This test case tests the working of OverSubscription
   feature which is part of HMM.*/

#include <hip_test_common.hh>
#ifdef __linux__
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#endif
#include <list>

#define INIT_VAL 2.5
#define NUM_ELMS 268435456  // 268435456 * 4 = 1GB
#define ITERATIONS 10
#define ONE_GB 1024 * 1024 * 1024

static void GetTotGpuMem(int *TotMem);
static void DisplayHmmFlgs(int *Signal);
// Kernel function
__global__ void Square(int n, float *x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = x[i] + 10;
  }
}

static void OneGBMemTest(int dev) {
  int DataMismatch = 0;
  float *HmmAG = nullptr;
  hipStream_t strm;
  HIP_CHECK(hipStreamCreate(&strm));
  // Testing hipMemAttachGlobal Flag
  HIP_CHECK(hipMallocManaged(&HmmAG, NUM_ELMS * sizeof(float),
                            hipMemAttachGlobal));

  // Initializing HmmAG memory
  for (int i = 0; i < NUM_ELMS; i++) {
    HmmAG[i] = INIT_VAL;
  }

  int blockSize = 256;
  int numBlocks = (NUM_ELMS + blockSize - 1) / blockSize;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(blockSize, 1, 1);
  HIP_CHECK(hipSetDevice(dev));
  for (int i = 0; i < ITERATIONS; ++i) {
    Square<<<dimGrid, dimBlock, 0, strm>>>(NUM_ELMS, HmmAG);
  }
  HIP_CHECK(hipStreamSynchronize(strm));
  for (int j = 0; j < NUM_ELMS; ++j) {
    if (HmmAG[j] != (INIT_VAL + ITERATIONS * 10)) {
      DataMismatch++;
      break;
    }
  }
  if (DataMismatch != 0) {
    WARN("Data Mismatch observed when kernel launched on device: " << dev);
    REQUIRE(false);
  }
  HIP_CHECK(hipFree(HmmAG));
  HIP_CHECK(hipStreamDestroy(strm));
}

static void GetTotGpuMem(int *TotMem) {
  size_t FreeMem, TotGpuMem;
  HIP_CHECK(hipMemGetInfo(&FreeMem, &TotGpuMem));
  TotMem[0] = (TotGpuMem/(ONE_GB));
  TotMem[1] = 1;
}

static void DisplayHmmFlgs(int *Signal) {
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

  // Checking for Vega20 or MI100
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  char *p = NULL;
  p = strstr(prop.gcnArchName, "gfx906");
  if (p) {
    WARN("This system has MI60 gpu hence OverSubscription test will be");
    WARN(" skipped");
    Signal[2] = 1;
  }
  p = strstr(prop.gcnArchName, "gfx908");
  if (p) {
    WARN("This system has MI100 gpu hence OverSubscription test will be");
    WARN(" skipped");
    Signal[2] = 1;
  }
  Signal[1] = managed;
  Signal[0] = 1;
}

TEST_CASE("Unit_HMM_OverSubscriptionTst") {
  int HmmEnabled = 0;
  // The following Shared Mem is to get Max GPU Mem
  // The size requested is for three ints
  // 1) To get Max GPU Mem in GB
  // 2) To Signal parent that req. info is available to consume
  // 3) To know if MI60 or MI100 gpu are there in the system
  key_t key = ftok("shmTotMem", 66);
  int shmid = shmget(key, (3 * sizeof(int)), 0666|IPC_CREAT);
  int *TotGpuMem = reinterpret_cast<int*>(shmat(shmid, NULL, 0));
  TotGpuMem[0] = 0; TotGpuMem[1] = 0;
  // The following function DisplayHmmFlgs() displays the flag values related
  // to HMM and also sends us ManagedMemory attribute value
  if (fork() == 0) {
    DisplayHmmFlgs(TotGpuMem);
    exit(1);
  }
  while (TotGpuMem[0] == 0) {
      sleep(2);
  }
  // The following if block will skip test if either of MI60 or MI100 is found
  if (TotGpuMem[2] == 1) {
    SUCCEED("Test is skipped!!");
    REQUIRE(true);
  } else {
    HmmEnabled = TotGpuMem[1];

    // Re-setting the shared memory values for further usage
    TotGpuMem[0] = 0;
    TotGpuMem[1] = 0;

    std::list<pid_t> PidLst;
    // The following function gets the MaxGpu memory in GBs and also launches
    // OverSubscription test
    if (HmmEnabled) {
      if ((setenv("HSA_XNACK", "1", 1)) != 0) {
        WARN("Unable to turn on HSA_XNACK, hence terminating the Test case!");
        REQUIRE(false);
      }
      if (fork() == 0) {
        GetTotGpuMem(TotGpuMem);
      }
      while (TotGpuMem[1] == 0) {
        sleep(2);
      }
      int NumGB = TotGpuMem[0], TotalThreads = (NumGB + 10);
      WARN("Launching " << TotalThreads);
      WARN(" processes to test OverSubscription.");
      pid_t pid;
      for (int k = 0; k < TotalThreads; ++k) {
        pid = fork();
        PidLst.push_back(pid);
        if (pid == 0) {
          OneGBMemTest(0);
          exit(10);
        }
      }
    } else {
      SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
              "attribute. Hence skipping the testing with Pass result.\n");
    }
    int status;
    for (pid_t pd : PidLst) {
      waitpid(pd, &status, 0);
      if (!(WIFEXITED(status))) {
        REQUIRE(false);
      }
    }
  }
  shmdt(TotGpuMem);
  shmctl(shmid, IPC_RMID, NULL);
}
