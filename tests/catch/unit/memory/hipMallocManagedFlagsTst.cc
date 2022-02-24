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

// Kernel function
__global__ void MallcMangdFlgTst(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * x[i];
}


// The following function prints info on attributes related to HMM
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

// The following section tests working of hipMallocManaged with flag parameters
TEST_CASE("Unit_hipMallocManaged_FlgParam") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    std::atomic<int> DataMismatch{0};
    bool IfTestPassed = true;
    float *HmmAG = NULL, *HmmAH1 = NULL, *HmmAH2 = NULL, INIT_VAL = 2.5;
    int NumDevs = 0, NUM_ELMS = 4096;
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    float *Ad = NULL, *Ah = NULL;
    Ah = new float[NUM_ELMS];
    // Testing hipMemAttachGlobal Flag
    HIP_CHECK(hipMallocManaged(&HmmAG, NUM_ELMS * sizeof(float),
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
    hipStream_t strm;
    for (int i = 0; i < NumDevs; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMalloc(&Ad, NUM_ELMS * sizeof(float)));
      HIP_CHECK(hipMemset(Ad, 0, NUM_ELMS * sizeof(float)));
      MallcMangdFlgTst<<<dimGrid, dimBlock, 0, strm>>>(NUM_ELMS, HmmAG, Ad);
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipMemcpy(Ah, Ad, NUM_ELMS * sizeof(float),
                         hipMemcpyDeviceToHost));
      for (int j = 0; j < NUM_ELMS; ++j) {
        if (Ah[j] != (INIT_VAL * INIT_VAL)) {
          DataMismatch++;
        }
      }
      if (DataMismatch != 0) {
        WARN("Data Mismatch observed when kernel launched on");
        WARN(" device: " << i);
        IfTestPassed = false;
      }
      DataMismatch = 0;

      HIP_CHECK(hipFree(Ad));
      HIP_CHECK(hipStreamDestroy(strm));
    }
    delete[] Ah;
    HIP_CHECK(hipFree(HmmAG));

    DataMismatch = 0;
    HIP_CHECK(hipMallocManaged(&HmmAH1, NUM_ELMS * sizeof(float),
                              hipMemAttachHost));
    HIP_CHECK(hipMallocManaged(&HmmAH2, NUM_ELMS * sizeof(float),
                              hipMemAttachHost));

    // Initializing HmmAH memory
    for (int i = 0; i < NUM_ELMS; i++) {
      HmmAH1[i] = INIT_VAL;
      HmmAH2[i] = 0;
    }
    for (int i = 0; i < NumDevs; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMemset(HmmAH2, 0, NUM_ELMS * sizeof(float)));
      MallcMangdFlgTst<<<dimGrid, dimBlock, 0, strm>>>(NUM_ELMS,
                                                       HmmAH1, HmmAH2);
      HIP_CHECK(hipStreamSynchronize(strm));
      for (int j = 0; j < NUM_ELMS; ++j) {
        if (HmmAH2[j] != (INIT_VAL * INIT_VAL)) {
          DataMismatch++;
        }
      }
      if (DataMismatch != 0) {
        WARN("Data Mismatch observed when kernel launched on");
        WARN(" device: " << i);
        IfTestPassed = false;
      }
      HIP_CHECK(hipStreamDestroy(strm));
    }
    HIP_CHECK(hipFree(HmmAH1));
    HIP_CHECK(hipFree(HmmAH2));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("Gpu doesnt support HMM! Hence skipping the test with PASS result");
  }
}

// The following function tests Memory access allocated using hipMallocManaged
// in multiple streams
TEST_CASE("Unit_hipMallocManaged_AccessMultiStream") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    std::atomic<int> DataMismatch{0};
    bool IfTestPassed = true;
    float *HmmAG = NULL, *HmmAH1 = NULL, *HmmAH2 = NULL, INIT_VAL = 2.5;
    int NumStrms = 0, MultiDevice = 0, NUM_ELMS = 4096;
    HIP_CHECK(hipGetDeviceCount(&MultiDevice));
    if (MultiDevice >= 2) {
      HIP_CHECK(hipGetDeviceCount(&NumStrms));
    } else {
      NumStrms = 4;
    }
    hipStream_t **Stream = new hipStream_t*[NumStrms];
    for (int i = 0; i < NumStrms; ++i) {
      Stream[i] = reinterpret_cast<hipStream_t*>(malloc(sizeof(hipStream_t)));
    }
    float *Ad = NULL, *Ah = NULL;
    Ah = new float[NUM_ELMS];
    for (int i = 0; i < NumStrms; ++i) {
      if (MultiDevice >= 2) {
        HIP_CHECK(hipSetDevice(i));
      }
      HIP_CHECK(hipStreamCreate(Stream[i]));
    }
    HIP_CHECK(hipSetDevice(0));
    // Testing hipMemAttachGlobal Flag
    HIP_CHECK(hipMallocManaged(&HmmAG, NUM_ELMS * sizeof(float),
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
    for (int i = 0; i < NumStrms; i++) {
      if (MultiDevice >= 2) {
        HIP_CHECK(hipSetDevice(i));
      }
      HIP_CHECK(hipMalloc(&Ad, NUM_ELMS * sizeof(float)));
      HIP_CHECK(hipMemset(Ad, 0, NUM_ELMS * sizeof(float)));
      MallcMangdFlgTst<<<dimGrid, dimBlock, 0, *(Stream[i])>>>(NUM_ELMS,
                                                            HmmAG, Ad);
      HIP_CHECK(hipStreamSynchronize(*(Stream[i])));
      // Validating the results
      HIP_CHECK(hipMemcpy(Ah, Ad, NUM_ELMS * sizeof(float),
                         hipMemcpyDeviceToHost));
      for (int j = 0; j < NUM_ELMS; ++j) {
        if (Ah[j] != (INIT_VAL * INIT_VAL)) {
          DataMismatch++;
        }
      }
      if (DataMismatch != 0) {
        WARN("Data Mismatch observed when kernel launched on");
        WARN(" device: " << i);
        IfTestPassed = false;
      }
      DataMismatch = 0;

      HIP_CHECK(hipFree(Ad));
    }
    delete[] Ah;
    HIP_CHECK(hipFree(HmmAG));

    DataMismatch = 0;
    HIP_CHECK(hipMallocManaged(&HmmAH1, NUM_ELMS * sizeof(float),
                                hipMemAttachHost));
    HIP_CHECK(hipMallocManaged(&HmmAH2, NUM_ELMS * sizeof(float),
                              hipMemAttachHost));

    // Initializing HmmAH memory
    for (int i = 0; i < NUM_ELMS; i++) {
      HmmAH1[i] = INIT_VAL;
      HmmAH2[i] = 0;
    }
    for (int i = 0; i < NumStrms; i++) {
      if (MultiDevice >= 2) {
        HIP_CHECK(hipSetDevice(i));
      }
      HIP_CHECK(hipMemset(HmmAH2, 0, NUM_ELMS * sizeof(float)));
      MallcMangdFlgTst<<<dimGrid, dimBlock, 0, *(Stream[i])>>>(NUM_ELMS,
                                                            HmmAH1, HmmAH2);
      HIP_CHECK(hipStreamSynchronize(*(Stream[i])));
      for (int j = 0; j < NUM_ELMS; ++j) {
        if (HmmAH2[j] != (INIT_VAL * INIT_VAL)) {
          DataMismatch++;
          break;
        }
      }
      if (DataMismatch != 0) {
        WARN("Data Mismatch observed when kernel launched on");
        WARN(" device: " << i);
        IfTestPassed = false;
      }
    }

    HIP_CHECK(hipFree(HmmAH1));
    HIP_CHECK(hipFree(HmmAH2));
    for (int i = 0; i < NumStrms; ++i) {
      HIP_CHECK(hipStreamDestroy(*(Stream[i])));
    }
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("Gpu doesnt support HMM! Hence skipping the test with PASS result");
  }
}

