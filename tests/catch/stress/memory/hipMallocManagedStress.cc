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
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

//  The following test case allocation, host access, device access of HMM
//   memory from size 1 to 10KB
/*  Test Case Description:
    1) Testing allocation, host access, device access of HMM
     memory from size 1 to 10KB
    2) The following test case tests the behavior of kernel with a HMM memory
        and hipMalloc memory
    3) The following test case tests when the same Hmm memory is used for
       launching multiple different kernels will results in any issue
    4) Testing the allocation of/scenarios around max possible memory
     */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define INCRMNT 10
// Kernel function
__global__ void KrnlWth2MemTypesC(unsigned char *Hmm, unsigned char *Dptr,
                                  size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Dptr[i] + INCRMNT;
  }
}
static bool IfTestPassed = true;

// Kernel functions
__global__ void KrnlWth2MemTypes(int *Hmm, int *Dptr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = index; i < n; i++) {
    Hmm[i] = Dptr[i] + 10;
  }
}

__global__ void KernelMulAdd_MngdMem(int *Hmm, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Hmm[i] * 2 + 10;
  }
}

__global__ void KernelMul_MngdMem(int *Hmm, int *Dptr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Dptr[i] * 10;
  }
}
static bool IfTestPassed = true;

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
//  The following test case allocation, host access, device access of HMM
//   memory from size 1 to 10KB

TEST_CASE("Stress_hipMallocManaged_MultiSize") {
  IfTestPassed = true;
  int managed = HmmAttrPrint();
  if (managed == 1) {
    unsigned char *Hmm1 = nullptr, *Hmm2 = nullptr;
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
        if (Hmm2[k] != (InitVal + INCRMNT)) {
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
      if (IfTestPassed == false) {
        HIP_CHECK(hipStreamDestroy(strm));
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipStreamDestroy(strm));
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

// The following test case tests the behavior of kernel with a HMM memory and
// hipMalloc memory

TEST_CASE("Stress_hipMallocManaged_KrnlWth2MemTypes") {
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
TEST_CASE("Stress_hipMallocManaged_MultiKrnlHmmAccess") {
  int managed = HmmAttrPrint();
  if (managed) {
    int InitVal = 123, NumElms = (1024 * 1024);
    LaunchKrnl4(NumElms, InitVal);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}

// Testing the allocation of/scenarios around max possible memory
TEST_CASE("Stress_hipMallocManaged_ExtremeSizes") {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    bool IfTestPassed = true;
    hipError_t err;
    void *Hmm = NULL;
    size_t totalDevMem = 0, freeDevMem = 0;
    int NumDevs = 0;
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    // Testing allocation of extreme and unusual mem values
    for (int i = 0; i < NumDevs; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipMemGetInfo(&freeDevMem, &totalDevMem));
      err = hipMallocManaged(&Hmm, 1, hipMemAttachGlobal);
      if (hipSuccess == err) {
        HIP_CHECK(hipFree(Hmm));
      } else {
        WARN("Observed error while allocating memory on GPU: " << i);
        WARN(" size 1 with");
        WARN(" hipMallocManaged() api with flag 'hipMemAttachGlobal'\n");
        WARN("Error: " << hipGetErrorString(err));
        IfTestPassed = false;
      }
      err = hipMallocManaged(&Hmm, freeDevMem, hipMemAttachGlobal);
      if (hipSuccess == err) {
        HIP_CHECK(hipFree(Hmm));
      } else {
        WARN("Observed error while allocating max free memory on GPU: " << i);
        WARN(" with hipMallocManaged() api with flag 'hipMemAttachGlobal'\n");
        WARN("Error: " << hipGetErrorString(err));
        IfTestPassed = false;
      }
      err = hipMallocManaged(&Hmm, (freeDevMem - 1), hipMemAttachGlobal);
      if (hipSuccess == err) {
        HIP_CHECK(hipFree(Hmm));
      } else {
        WARN("Observed error while allocating max (free - 1) memory on ");
        WARN("GPU: " << i);
        WARN(" using hipMallocManaged() api with flag 'hipMemAttachGlobal'\n");
        WARN("Error: " << hipGetErrorString(err));
        IfTestPassed = false;
      }
      err = hipMallocManaged(&Hmm, 1, hipMemAttachHost);
      if (hipSuccess == err) {
        HIP_CHECK(hipFree(Hmm));
      } else {
        WARN("Observed error while allocating memory size 1 on GPU: " << i);
        WARN(" with hipMallocManaged() api with flag 'hipMemAttachHost'\n");
        WARN("Error: " << hipGetErrorString(err));
        IfTestPassed = false;
      }
      err = hipMallocManaged(&Hmm, freeDevMem, hipMemAttachHost);
      if (hipSuccess == err) {
        HIP_CHECK(hipFree(Hmm));
      } else {
        WARN("Observed error while allocating max free memory on GPU: " << i);
        WARN(" with hipMallocManaged() api with flag 'hipMemAttachHost'\n");
        WARN("Error: " << hipGetErrorString(err));
        IfTestPassed = false;
      }
      err = hipMallocManaged(&Hmm, (freeDevMem - 1), hipMemAttachHost);
      if (hipSuccess == err) {
        HIP_CHECK(hipFree(Hmm));
      } else {
        WARN("Observed error while allocating max (freeDevMem - 1) memory"
               " on GPU: " << i);
        WARN(" with hipMallocManaged() api with flag 'hipMemAttachHost'\n");
        WARN("Error: " << hipGetErrorString(err));
        IfTestPassed = false;
      }
    }
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("Gpu doesnt support HMM! Hence skipping the test with PASS result");
  }
}
