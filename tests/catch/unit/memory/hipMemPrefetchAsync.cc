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
// Kernel function
__global__ void MemPrftchAsyncKernel(int* C_d, const int* A_d, size_t N) {
  size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
  size_t stride = hipBlockDim_x * hipGridDim_x;
  for (size_t i = offset; i < N; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
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

/*
  Test Description: This test prefetches the memory to each of the available
  devices and launch kernel followed by result verification
  At the end the memory is prefetched to Host and kernel is launched followed
  by result verification.
*/

TEST_CASE("Unit_hipMemPrefetchAsync") {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int A_CONST = 123, MEM_SIZE = (8192 * sizeof(int));
    int *devPtr1 = NULL, *devPtr2 = NULL, NumDevs = 0, flag = 0;
    hipStream_t strm;
    HIP_CHECK(hipMallocManaged(&devPtr1, MEM_SIZE));
    HIP_CHECK(hipMallocManaged(&devPtr2, MEM_SIZE));
    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    // Initializing the memory
    for (uint32_t k = 0; k < (MEM_SIZE/sizeof(int)); ++k) {
      devPtr1[k] = A_CONST;
      devPtr2[k] = 0;
    }


    for (int i = 0; i < NumDevs;  ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMemPrefetchAsync(devPtr1, MEM_SIZE, i, strm));
      HIP_CHECK(hipStreamSynchronize(strm));
      MemPrftchAsyncKernel<<<32, (MEM_SIZE/sizeof(int)/32)>>>(devPtr2, devPtr1,
                                                       MEM_SIZE/sizeof(int));
      for (uint32_t m = 0; m < (MEM_SIZE/sizeof(int)); ++m) {
        if (devPtr1[m] != (A_CONST * A_CONST)) {
          flag = 1;
        }
      }
      HIP_CHECK(hipStreamDestroy(strm));
      if (!flag) {
        INFO("Test failed for device: " << i);
        IfTestPassed = false;
        flag = 0;
      }
    }
    // The memory will be prefetched from last gpu in the system to the host
    // memory and kernel is launched followed by result verification.
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipMemPrefetchAsync(devPtr1, MEM_SIZE, hipCpuDeviceId, strm));
    HIP_CHECK(hipStreamSynchronize(strm));
    MemPrftchAsyncKernel<<<32, (MEM_SIZE/sizeof(int)/32)>>>(devPtr2, devPtr1,
                                                     MEM_SIZE/sizeof(int));
    for (uint32_t m = 0; m < (MEM_SIZE/sizeof(int)); ++m) {
      if (devPtr1[m] != (A_CONST * A_CONST)) {
        flag = 1;
      }
    }
    HIP_CHECK(hipStreamDestroy(strm));
    if (!flag) {
      INFO("Failed to prefetch the memory to System space.\n");
      IfTestPassed = false;
      flag = 0;
    }

    HIP_CHECK(hipFree(devPtr1));
    HIP_CHECK(hipFree(devPtr2));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
           "attribute. Hence skipping the testing with Pass result.\n");
  }
}
