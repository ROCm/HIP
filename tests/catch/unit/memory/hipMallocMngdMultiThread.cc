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
#include <atomic>


// Kernel functions
__global__ void HmmMultiThread(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * x[i];
}

__global__ void KrnlWth2MemTypes(int *Hmm, int *Dptr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = index; i < n; i++) {
    Hmm[i] = Dptr[i] + 10;
  }
}

__global__ void KernelMul_MngdMem123(int *Hmm, int *Dptr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Dptr[i] * 10;
  }
}



// The following variable is used to determine the failure of test case
static bool  IfTestPassed = true;

static void LaunchKrnl(int *Hmm1, size_t NumElms, int InitVal, int GpuOrdnl,
                       int AdviseFlg) {
  int *Hmm2 = NULL;
  hipStream_t strm;
  HIPCHECK(hipSetDevice(GpuOrdnl));
  HIPCHECK(hipStreamCreate(&strm));
  if (AdviseFlg == 0) {
    HIPCHECK(hipMemAdvise(Hmm1 , NumElms * sizeof(int),
                           hipMemAdviseSetReadMostly, GpuOrdnl));
  } else if (AdviseFlg == 1) {
    HIPCHECK(hipMemAdvise(Hmm1 , NumElms * sizeof(int),
                           hipMemAdviseSetPreferredLocation, GpuOrdnl));
  } else if (AdviseFlg == 2) {
    HIPCHECK(hipMemAdvise(Hmm1 , NumElms * sizeof(int),
                           hipMemAdviseSetAccessedBy, GpuOrdnl));
  } else if (AdviseFlg == 3) {
    HIPCHECK(hipMemPrefetchAsync(Hmm1, NumElms * sizeof(int), GpuOrdnl, strm));
    HIPCHECK(hipStreamSynchronize(strm));
  }
  HIPCHECK(hipMallocManaged(&Hmm2, (sizeof(int) * NumElms)));
  for (int i = 0; i < 2; ++i) {
    KrnlWth2MemTypes<<<((NumElms + 63)/64), 64, 0, strm>>>(Hmm2, Hmm1, NumElms);
    HIPCHECK(hipStreamSynchronize(strm));
  }
  // Verifying the result
  int DataMismatch = 0;
  for (size_t i = 0; i < NumElms; ++i) {
    if (Hmm2[i] != (InitVal + 10)) {
      DataMismatch++;
    }
  }
  if (DataMismatch != 0) {
    WARN("Data Mismatch observed at line: " << __LINE__);
    IfTestPassed = false;
  }
}

static void LaunchKrnl2(int *Hmm, size_t NumElms, int InitVal, int HmmMem) {
  int *ptr = nullptr, blockSize = 64, *HstPtr = nullptr;
  hipStream_t strm;
  HIPCHECK(hipStreamCreate(&strm));
  if (HmmMem == 0) {
    HstPtr = reinterpret_cast<int*>(new int[NumElms]);
    HIPCHECK(hipMalloc(&ptr, (sizeof(int) * NumElms)));
  } else {
    HIPCHECK(hipMallocManaged(&ptr, (sizeof(int) * NumElms)));
  }
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
  for (int i = 0; i < 2; ++i) {
    KrnlWth2MemTypes<<<dimGrid, dimBlock, 0, strm>>>(ptr, Hmm, NumElms);
  }
  HIPCHECK(hipStreamSynchronize(strm));
  // Verifying the result
  int DataMismatch = 0;
  if (HmmMem == 0) {
    HIPCHECK(hipMemcpy(HstPtr, ptr, (sizeof(int) * NumElms),
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < NumElms; ++i) {
      if (HstPtr[i] != (InitVal + 10)) {
        DataMismatch++;
      }
    }
  } else {
    for (size_t i = 0; i < NumElms; ++i) {
      if (ptr[i] != (InitVal + 10)) {
        DataMismatch++;
      }
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observed at line: " << __LINE__);
    REQUIRE(false);
  }
}

static void LaunchKrnl3(int *Dptr, size_t NumElms, int InitVal) {
  int *Hmm = NULL, blockSize = 64;
  hipStream_t strm;
  HIPCHECK(hipStreamCreate(&strm));
  HIPCHECK(hipMallocManaged(&Hmm, (sizeof(int) * NumElms)));
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
  for (int i = 0; i < 2; ++i) {
    KrnlWth2MemTypes<<<dimGrid, dimBlock, 0, strm>>>(Hmm, Dptr, NumElms);
  }
  HIPCHECK(hipStreamSynchronize(strm));
  // Verifying the result
  int DataMismatch = 0;
  for (size_t i = 0; i < NumElms; ++i) {
    if (Hmm[i] != (InitVal + 10)) {
      DataMismatch++;
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observed at line: " << __LINE__);
    REQUIRE(false);
  }
}


static void LaunchKrnl5(int *Hmm1, size_t NumElms, int InitVal,
                        int KerneltoLaunch) {
  int *Hmm2 = NULL, blockSize = 64;
  hipStream_t strm;
  HIPCHECK(hipStreamCreate(&strm));
  HIPCHECK(hipMallocManaged(&Hmm2, (sizeof(int) * NumElms)));
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
  for (int i = 0; i < 2; ++i) {
    if (KerneltoLaunch == 0) {
      KrnlWth2MemTypes<<<dimGrid, dimBlock, 0, strm>>>(Hmm2, Hmm1, NumElms);
    } else {
      KernelMul_MngdMem123<<<dimGrid, dimBlock, 0, strm>>>(Hmm2, Hmm1, NumElms);
    }
  }
  HIPCHECK(hipStreamSynchronize(strm));
  // Verifying the result
  int DataMismatch = 0;
  if (KerneltoLaunch == 0) {
    for (size_t i = 0; i < NumElms; ++i) {
      if (Hmm2[i] != (InitVal + 10)) {
        DataMismatch++;
      }
    }
  } else {
    for (size_t i = 0; i < NumElms; ++i) {
      if (Hmm2[i] != (InitVal * 10)) {
        DataMismatch++;
      }
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observed at line: " << __LINE__);
    REQUIRE(false);
  }
}


static void TestFlagParamGlobal(int dev) {
  std::atomic<int> DataMismatch{0};
  int NUM_ELMS = 4096, ITERATIONS = 10;
  float *HmmAG = NULL, INIT_VAL = 2.5;
  float *Ad = NULL, *Ah = NULL;
  Ah = new float[NUM_ELMS];
  hipStream_t strm;
  HIPCHECK(hipSetDevice(dev));
  HIPCHECK(hipStreamCreate(&strm));
  // Testing hipMemAttachGlobal Flag
  HIPCHECK(hipMallocManaged(&HmmAG, NUM_ELMS * sizeof(float),
                            hipMemAttachGlobal));

  // Initializing HmmAG memory
  for (int i = 0; i < NUM_ELMS; i++) {
    HmmAG[i] = INIT_VAL;
    Ah[i] = 0;
  }

  int blockSize = 256;
  int numBlocks = (NUM_ELMS + blockSize - 1) / blockSize;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(blockSize, 1, 1);
  HIPCHECK(hipSetDevice(dev));
  HIPCHECK(hipMalloc(&Ad, NUM_ELMS * sizeof(float)));
  HIPCHECK(hipMemset(Ad, 0, NUM_ELMS * sizeof(float)));
  for (int i = 0; i < ITERATIONS; ++i) {
    HmmMultiThread<<<dimGrid, dimBlock, 0, strm>>>(NUM_ELMS, HmmAG, Ad);
    HIPCHECK(hipStreamSynchronize(strm));
  }
  HIPCHECK(hipMemcpy(Ah, Ad, NUM_ELMS * sizeof(float), hipMemcpyDeviceToHost));
  for (int j = 0; j < NUM_ELMS; ++j) {
    if (Ah[j] != (INIT_VAL * INIT_VAL)) {
      DataMismatch++;
      break;
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observed when kernel launched on device: " << dev);
    IfTestPassed = false;
  }

  HIPCHECK(hipFree(Ad));
  delete[] Ah;
  HIPCHECK(hipFree(HmmAG));
  HIPCHECK(hipStreamDestroy(strm));
}


static void TestFlagParamHost(int dev) {
  std::atomic<int> DataMismatch{0};
  float *HmmAH1 = nullptr, *HmmAH2 = nullptr, INIT_VAL = 2.5;
  int NUM_ELMS = 4096, ITERATIONS = 10;
  hipStream_t strm;
  HIPCHECK(hipSetDevice(dev));
  HIPCHECK(hipStreamCreate(&strm));
  HIPCHECK(hipMallocManaged(&HmmAH1, NUM_ELMS * sizeof(float),
                            hipMemAttachHost));
  HIPCHECK(hipMallocManaged(&HmmAH2, NUM_ELMS * sizeof(float),
                            hipMemAttachHost));
  // Initializing HmmAH memory
  for (int i = 0; i < NUM_ELMS; i++) {
    HmmAH1[i] = INIT_VAL;
    HmmAH2[i] = 0;
  }
  int blockSize = 256;
  int numBlocks = (NUM_ELMS + blockSize - 1) / blockSize;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(blockSize, 1, 1);
  for (int i = 0; i < ITERATIONS; ++i) {
    HmmMultiThread<<<dimGrid, dimBlock, 0, strm>>>(NUM_ELMS, HmmAH1, HmmAH2);
    HIPCHECK(hipStreamSynchronize(strm));
  }
  for (int j = 0; j < NUM_ELMS; ++j) {
    if (HmmAH2[j] != (INIT_VAL * INIT_VAL)) {
      IfTestPassed = false;
      DataMismatch++;
      break;
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observed when kernel launched on device: " << dev);
    IfTestPassed = false;
  }
  HIPCHECK(hipFree(HmmAH1));
  HIPCHECK(hipFree(HmmAH2));
  HIPCHECK(hipStreamDestroy(strm));
}

static void AllocateHmmMemory(int flag, int device) {
  int ITERATIONS = 10;
  void *HmmAG = NULL, *HmmAH = NULL;
  HIPCHECK(hipSetDevice(device));
  for (int i = 0; i < ITERATIONS; ++i) {
    if (!flag) {
      HIPCHECK(hipMallocManaged(&HmmAG, (2 * 4096), hipMemAttachGlobal));
      HIPCHECK(hipFree(HmmAG));
    } else {
      HIPCHECK(hipMallocManaged(&HmmAH, (2 * 4096), hipMemAttachHost));
      HIPCHECK(hipFree(HmmAH));
    }
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

TEST_CASE("Unit_hipMallocManaged_MultiThread") {
  IfTestPassed = true;
  int NumDevs = 0, managed = 0, ATTACH_GLOBAL = 0, ATTACH_HOST = 1;
  int ITERATIONS = 10;
  managed = HmmAttrPrint();
  if (managed) {
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    std::vector<std::thread> T1;
    std::vector<std::thread> T2;
    for (int i = 0; i < NumDevs; ++i) {
      for (int j = 0; j < ITERATIONS; ++j) {
        T1.push_back(std::thread(TestFlagParamGlobal, i));
        T2.push_back(std::thread(AllocateHmmMemory, ATTACH_GLOBAL, i));
      }
      for (auto &t1 : T1) {
        if (t1.joinable()) {
          t1.join();
        }
      }
      for (auto &t2 : T2) {
        if (t2.joinable()) {
          t2.join();
        }
      }
    }
    T1.clear();
    T2.clear();
    for (int i = 0; i < NumDevs; ++i) {
      for (int j = 0; j < ITERATIONS; ++j) {
        T1.push_back(std::thread(TestFlagParamHost, i));
        T2.push_back(std::thread(AllocateHmmMemory, ATTACH_HOST, i));
      }
      for (auto &t1 : T1) {
        if (t1.joinable()) {
          t1.join();
        }
      }
      for (auto &t2 : T2) {
        if (t2.joinable()) {
          t2.join();
        }
      }
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory"
            "attribute. Hence skipping the testing with Pass result.\n");
  }
  REQUIRE(IfTestPassed);
}

// The following test checks what happens when same Hmm memory is used to
// launch multiple threads over multiple gpus
TEST_CASE("Unit_hipMallocManaged_MGpuMThread") {
  IfTestPassed = true;
  int Ngpus = 0;
  HIP_CHECK(hipGetDeviceCount(&Ngpus));
  if (Ngpus < 2) {
    WARN("This test needs atleast 2 or more gpus, but the system");
    WARN(" has only " << Ngpus);
    WARN(" gpus. Hence skipping the test.");
    SUCCEED("\n");
  }
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int InitVal = 123, *Hmm1 = NULL, NumElms = 4096*4;
    HIP_CHECK(hipMallocManaged(&Hmm1, (NumElms * sizeof(int))));
    for (int i = 0; i < NumElms; ++i) {
      Hmm1[i] = InitVal;
    }

    std::vector<std::thread> Thrds;
    // AdviseFlg=0 for ReadMostly to be applied
    // AdviseFlg=1 for PreferredLocation to be applied
    // AdviseFlg=2 for AccessedBy to be applied
    // AdviseFlg=3 to prefetch the memory to particular gpu
    for (int AdviseFlg = 0; AdviseFlg < 4; ++AdviseFlg) {
      for (int i = 0; i < Ngpus; ++i) {
        Thrds.push_back(std::thread(LaunchKrnl, Hmm1, NumElms, InitVal, i,
                                    AdviseFlg));
      }
      for (auto &thr : Thrds) {
        if (thr.joinable()) {
          thr.join();
        }
      }
    }
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


// The following test checks what happens when multiple kernels are launched
// with same Hmm memory
TEST_CASE("Unit_hipMallocManaged_MultiKrnlComnHmm") {
  IfTestPassed = true;
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int InitVal = 123, *Hmm = NULL, NumElms = 1024*4, TotThrds = 2;
    int HmmMem2 = 0, *HstPtr = nullptr;  //  to indicate the thread that
    //  hipMalloc() memory has to be used
    HstPtr = reinterpret_cast<int*>(new int[NumElms]);
    HIP_CHECK(hipMalloc(&Hmm, (NumElms * sizeof(int))));
    for (int i = 0; i < NumElms; ++i) {
      HstPtr[i] = InitVal;
    }
    HIP_CHECK(hipMemcpy(Hmm, HstPtr, (NumElms * sizeof(int)),
                        hipMemcpyHostToDevice));
    std::vector<std::thread> Thrds;
    for (int i = 0; i < TotThrds; ++i) {
      Thrds.push_back(std::thread(LaunchKrnl2, Hmm, NumElms, InitVal, HmmMem2));
    }

    for (auto &thr : Thrds) {
      if (thr.joinable()) {
        thr.join();
      }
    }
    delete[] HstPtr;
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


// The following test checks what happens when multiple kernels are launched
// with same hipMalloc() memory
TEST_CASE("Unit_hipMallocManaged_MultiKrnlComnMalloc") {
  IfTestPassed = true;
  int managed = HmmAttrPrint();
  if (managed) {
    int InitVal = 123, *Dptr = NULL, NumElms = 4096*8, TotThrds = 2;
    int *HstPtr = reinterpret_cast<int*>(new int[NumElms]);
    HIP_CHECK(hipMalloc(&Dptr, (NumElms * sizeof(int))));
    for (int i = 0; i < NumElms; ++i) {
      HstPtr[i] = InitVal;
    }
    HIP_CHECK(hipMemcpy(Dptr, HstPtr, (NumElms * sizeof(int)),
                        hipMemcpyHostToDevice));
    std::vector<std::thread> Thrds;
    for (int i = 0; i < TotThrds; ++i) {
      Thrds.push_back(std::thread(LaunchKrnl3, Dptr, NumElms, InitVal));
    }

    for (auto &thr : Thrds) {
      if (thr.joinable()) {
        thr.join();
      }
    }
    delete[] HstPtr;
    HIP_CHECK(hipFree(Dptr));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

//  The following section tests the scenario wherein multiple threads use their
//  own stream to launch kernel on common Hmm memory
TEST_CASE("Unit_hipMallocManaged_MultiThrdMultiStrm") {
  IfTestPassed = true;
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int NumElms = 4096*4;
    int *Hmm1 = NULL, TotlThrds = 4, InitVal = 123;
    int HmmMem = 1;  //  to indicate the thread that Hmm memory need to be
    //  used inside it
    HIP_CHECK(hipMallocManaged(&Hmm1, (NumElms * sizeof(int))));
    for (int i = 0; i < NumElms; ++i) {
      Hmm1[i] = InitVal;
    }
    std::vector<std::thread> Thrds;
    for (int i = 0; i < TotlThrds; ++i) {
      Thrds.push_back(std::thread(LaunchKrnl2, Hmm1, NumElms, InitVal, HmmMem));
    }

    for (auto &thr : Thrds) {
      if (thr.joinable()) {
        thr.join();
      }
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}



//  The following section tests the scenario wherein two threads each use
//  different kernel but common HMM memory
TEST_CASE("Unit_hipMallocManaged_TwoKrnlsComnHmmMem") {
  IfTestPassed = true;
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int InitVal = 123, *Dptr = NULL, NumElms = 4096*4, TotThrds = 2;
    int *HstPtr = reinterpret_cast<int*>(new int[NumElms]);
    HIP_CHECK(hipMalloc(&Dptr, (NumElms * sizeof(int))));
    for (int i = 0; i < NumElms; ++i) {
      HstPtr[i] = InitVal;
    }
    HIP_CHECK(hipMemcpy(Dptr, HstPtr, (NumElms * sizeof(int)),
                        hipMemcpyHostToDevice));
    std::vector<std::thread> Thrds;
    for (int i = 0; i < TotThrds; ++i) {
      Thrds.push_back(std::thread(LaunchKrnl5, Dptr, NumElms, InitVal, i));
    }

    for (auto &thr : Thrds) {
      if (thr.joinable()) {
        thr.join();
      }
    }
    delete[] HstPtr;
    HIP_CHECK(hipFree(Dptr));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


