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

/*
  List of Test cases:
  1)  Unit_hipMallocManaged_Basic
  2) Unit_hipMallocManaged_MultiSize
  3) Unit_hipMallocManaged_MultiKrnlHmmAccess
  4) Unit_hipMallocManaged_KrnlWth2MemTypes
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>



// Kernel functions
__global__ void KrnlWth2MemTypes(int *Hmm, int *Dptr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = index; i < n; i++) {
    Hmm[i] = Dptr[i] + 10;
  }
}

__global__ void KernelMul_MngdMem(int *Hmm, int *Dptr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Dptr[i] * 10;
  }
}

__global__ void KernelMulAdd_MngdMem(int *Hmm, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Hmm[i] * 2 + 10;
  }
}

__global__ void KrnlWth2MemTypesC(unsigned char *Hmm, unsigned char *Dptr,
                                  size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Dptr[i] + 10;
  }
}

// The following variable will be used to get the result of computation
// from multiple threads
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


static void LaunchKrnl4(size_t NumElms, int InitVal) {
  int *Hmm = NULL, *Dptr = NULL, blockSize = 64, DataMismatch = 0;
  hipStream_t strm;
  HIP_CHECK(hipStreamCreate(&strm));
  HIP_CHECK(hipMallocManaged(&Hmm, (sizeof(int) * NumElms)));
  HIP_CHECK(hipMalloc(&Dptr, (sizeof(int) * NumElms)));
  int *Hstptr = reinterpret_cast<int*>(new int[NumElms]);
  for (size_t i = 0; i < NumElms; ++i) {
    Hstptr[i] = InitVal;
  }
  HIP_CHECK(hipMemcpy(Dptr, Hstptr, (NumElms * sizeof(int)),
                      hipMemcpyHostToDevice));
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
  KrnlWth2MemTypes<<<dimGrid, dimBlock, 0, strm>>>(Hmm, Dptr, NumElms);
  HIP_CHECK(hipStreamSynchronize(strm));
  for (size_t i = 0; i < NumElms; ++i) {
    if (Hmm[i] != (InitVal + 10)) {
      DataMismatch++;
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observed after the Kernel: KrnlWth2MemTypes!!\n");
    REQUIRE(false);
  }
  DataMismatch = 0;
  KernelMul_MngdMem<<<dimGrid, dimBlock, 0, strm>>>(Hmm, Dptr, NumElms);
  HIP_CHECK(hipStreamSynchronize(strm));
  // Verifying the result
  for (size_t i = 0; i < NumElms; ++i) {
    if (Hmm[i] != (InitVal * 10)) {
      DataMismatch++;
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observedafter the Kernel: KernelMul_MngdMem!!\n");
    REQUIRE(false);
  }
  DataMismatch = 0;
  KernelMulAdd_MngdMem<<<dimGrid, dimBlock, 0, strm>>>(Hmm, NumElms);
  HIP_CHECK(hipStreamSynchronize(strm));
  // Verifying the result

  for (size_t i = 0; i < NumElms; ++i) {
    if (Hmm[i] != (InitVal * 10 * 2 + 10)) {
      DataMismatch++;
    }
  }
  if (DataMismatch != 0) {
    INFO("Data Mismatch observedafter the Kernel: KernelMul_MngdMem!!\n");
    REQUIRE(false);
  }
  delete[] Hstptr;
}



static size_t N{4 * 1024 * 1024};
static unsigned blocksPerCU{6};
static unsigned threadsPerBlock{256};

/*
   This testcase verifies the hipMallocManaged basic scenario - supported on all devices
 */

TEST_CASE("Unit_hipMallocManaged_Basic") {
    int numElements = (N < (64 * 1024 * 1024)) ? 64 * 1024 * 1024 : N;
    float *A, *B, *C;

    HIP_CHECK(hipMallocManaged(&A, numElements*sizeof(float)));
    HIP_CHECK(hipMallocManaged(&B, numElements*sizeof(float)));
    HIP_CHECK(hipMallocManaged(&C, numElements*sizeof(float)));
}

/*
   This testcase verifies the hipMallocManaged basic scenario - supported only on HMM enabled devices
 */

TEST_CASE("Unit_hipMallocManaged_Advanced") {
  int managed =  HmmAttrPrint();
  if (managed == 1) {
    int numElements = (N < (64 * 1024 * 1024)) ? 64 * 1024 * 1024 : N;
    float *A, *B, *C;

    HIP_CHECK(hipMallocManaged(&A, numElements*sizeof(float)));
    HIP_CHECK(hipMallocManaged(&B, numElements*sizeof(float)));
    HIP_CHECK(hipMallocManaged(&C, numElements*sizeof(float)));
    HipTest::setDefaultData(numElements, A, B, C);

    hipDevice_t device = hipCpuDeviceId;

    HIP_CHECK(hipMemAdvise(A, numElements*sizeof(float),
          hipMemAdviseSetReadMostly, device));
    HIP_CHECK(hipMemPrefetchAsync(A, numElements*sizeof(float), 0));
    HIP_CHECK(hipMemPrefetchAsync(B, numElements*sizeof(float), 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemRangeGetAttribute(&device, sizeof(device),
          hipMemRangeAttributeLastPrefetchLocation,
          A, numElements*sizeof(float)));
    if (device != 0) {
      INFO("hipMemRangeGetAttribute error, device = " << device);
    }
    uint32_t read_only = 0xf;
    HIP_CHECK(hipMemRangeGetAttribute(&read_only, sizeof(read_only),
          hipMemRangeAttributeReadMostly,
          A, numElements*sizeof(float)));
    if (read_only != 1) {
      SUCCEED("hipMemRangeGetAttribute error, read_only = " << read_only);
    }

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock,
                                            numElements);
    hipEvent_t event0, event1;
    HIP_CHECK(hipEventCreate(&event0));
    HIP_CHECK(hipEventCreate(&event1));
    HIP_CHECK(hipEventRecord(event0, 0));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
        0, 0, static_cast<const float*>(A),
        static_cast<const float*>(B), C, numElements);
    HIP_CHECK(hipEventRecord(event1, 0));
    HIP_CHECK(hipDeviceSynchronize());
    float time = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&time, event0, event1));
    printf("Time %.3f ms\n", time);
    float maxError = 0.0f;
    HIP_CHECK(hipMemPrefetchAsync(B, numElements*sizeof(float),
                                  hipCpuDeviceId));
    HIP_CHECK(hipDeviceSynchronize());
    device = 0;
    HIP_CHECK(hipMemRangeGetAttribute(&device, sizeof(device),
          hipMemRangeAttributeLastPrefetchLocation,
          A, numElements*sizeof(float)));
    if (device != hipCpuDeviceId) {
      SUCCEED("hipMemRangeGetAttribute error device = " << device);
    }

    for (int i = 0; i < numElements; i++) {
      maxError = fmax(maxError, fabs(B[i]-3.0f));
    }
    HIP_CHECK(hipFree(A));
    HIP_CHECK(hipFree(B));
    REQUIRE(maxError != 0.0f);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


// The following test case tests the behavior of kernel with a HMM memory and
// hipMalloc memory

TEST_CASE("Unit_hipMallocManaged_KrnlWth2MemTypes") {
  IfTestPassed = true;
  int *Hmm = NULL, *Dptr = NULL, InitVal = 123;
  size_t NumElms = (1024 * 1024);
  int *Hptr = new int[NumElms], blockSize = 64, DataMismatch = 0;
  int managed =  HmmAttrPrint();
  if (managed == 1) {
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipMallocManaged(&Hmm, sizeof(int) * NumElms));
    HIP_CHECK(hipMalloc(&Dptr, sizeof(int) * NumElms));
    for (size_t i = 0; i < NumElms; ++i) {
      Hmm[i] = 0;
      Hptr[i] = InitVal;
    }
    HIP_CHECK(hipMemcpy(Dptr, Hptr, sizeof(int) * NumElms,
                        hipMemcpyHostToDevice));
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid((NumElms + blockSize -1)/blockSize, 1, 1);
    KrnlWth2MemTypes<<<dimGrid, dimBlock, 0, strm>>>(Hmm, Dptr, NumElms);
    HIP_CHECK(hipStreamSynchronize(strm));
    // Verifying the results
    for (size_t k = 0; k < NumElms; ++k) {
      if (Hmm[k] != (InitVal + 10)) {
        DataMismatch++;
      }
    }
    if (DataMismatch != 0) {
      WARN("DataMismatch observed!\n");
      IfTestPassed = false;
    }

    HIP_CHECK(hipFree(Hmm));
    HIP_CHECK(hipFree(Dptr));
    delete[] Hptr;
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

// The following test case tests when the same Hmm memory is used for
// launching multiple different kernels will results in any issue
TEST_CASE("Unit_hipMallocManaged_MultiKrnlHmmAccess") {
  int managed = HmmAttrPrint();
  if (managed) {
    int InitVal = 123, NumElms = (1024 * 1024);
    LaunchKrnl4(NumElms, InitVal);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}


//  The following test case allocation, host access, device access of HMM
//   memory from size 1 to 10KB

TEST_CASE("Unit_hipMallocManaged_MultiSize") {
  IfTestPassed = true;
  int managed = HmmAttrPrint();
  if (managed == 1) {
    unsigned char *Hmm1 = NULL, *Hmm2 = NULL;
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
        if (Hmm2[k] != (InitVal + 10)) {
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
      REQUIRE(IfTestPassed);
    }
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
