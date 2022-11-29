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

/*
This testfile verifies the following scenarios of hipHostRegister API
1. Referencing the hipHostRegister variable from kernel and performing
   memset on that variable.This is verified for different datatypes.
2. hipHostRegister and perform hipMemcpy on it.
*/

#include "hip/hip_runtime_api.h"
#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>

#define OFFSET 128
static constexpr auto LEN{1024 * 1024};

template <typename T> __global__ void Inc(T* Ad) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tx] = Ad[tx] + static_cast<T>(1);
}

template <typename T>
void doMemCopy(size_t numElements, int offset, T* A, T* Bh, T* Bd, bool internalRegister) {
  constexpr auto memsetval = 13.0f;
  A = A + offset;
  numElements -= offset;

  size_t sizeBytes = numElements * sizeof(T);

  if (internalRegister) {
    HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
  }

  // Reset
  for (size_t i = 0; i < numElements; i++) {
    A[i] = static_cast<float>(i);
    Bh[i] = 0.0f;
  }

  HIP_CHECK(hipMemset(Bd, memsetval, sizeBytes));

  HIP_CHECK(hipMemcpy(Bd, A, sizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bh, Bd, sizeBytes, hipMemcpyDeviceToHost));

  // Make sure the copy worked
  ArrayMismatch(A, Bh, numElements);

  if (internalRegister) {
    HIP_CHECK(hipHostUnregister(A));
  }
}

/*
This testcase verifies the hipHostRegister API by
1. Allocating the memory using malloc
2. hipHostRegister that variable
3. Getting the corresponding device pointer of the registered varible
4. Launching kernel and access the device pointer variable
5. performing hipMemset on the device pointer variable
*/
TEMPLATE_TEST_CASE("Unit_hipHostRegister_ReferenceFromKernelandhipMemset", "", int, float, double) {
  size_t sizeBytes{LEN * sizeof(TestType)};
  TestType *A, **Ad;
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  Ad = new TestType*[num_devices];
  A = reinterpret_cast<TestType*>(malloc(sizeBytes));
  HIP_CHECK(hipHostRegister(A, sizeBytes, 0));

  for (int i = 0; i < LEN; i++) {
    A[i] = static_cast<TestType>(1);
  }

  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Ad[i]), A, 0));
  }

  // Reference the registered device pointer Ad from inside the kernel:
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    hipLaunchKernelGGL(Inc, dim3(LEN / 32), dim3(32), 0, 0, Ad[i]);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }
  REQUIRE(A[10] == 1 + static_cast<TestType>(num_devices));
  // Reference the registered device pointer Ad in hipMemset:
  for (int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemset(Ad[i], 0, sizeBytes));
  }
  REQUIRE(A[10] == 0);

  HIP_CHECK(hipHostUnregister(A));

  free(A);
  delete[] Ad;
}

/*
This testcase verifies hipHostRegister API by
performing memcpy on the hipHostRegistered variable.
*/
TEMPLATE_TEST_CASE("Unit_hipHostRegister_Memcpy", "", int, float, double) {
  // 1 refers to hipHostRegister
  // 0 refers to malloc
  auto mem_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));


  size_t sizeBytes = LEN * sizeof(TestType);
  TestType* A = reinterpret_cast<TestType*>(malloc(sizeBytes));

  // Copy to B, this should be optimal pinned malloc copy:
  // Note we are using the host pointer here:
  TestType *Bh, *Bd;
  Bh = reinterpret_cast<TestType*>(malloc(sizeBytes));
  HIP_CHECK(hipMalloc(&Bd, sizeBytes));

  REQUIRE(LEN > OFFSET);
  if (mem_type) {
    for (size_t i = 0; i < OFFSET; i++) {
      doMemCopy<TestType>(LEN, i, A, Bh, Bd, true /*internalRegister*/);
    }
  } else {
    HIP_CHECK(hipHostRegister(A, sizeBytes, 0));
    for (size_t i = 0; i < OFFSET; i++) {
      doMemCopy<TestType>(LEN, i, A, Bh, Bd, false /*internalRegister*/);
    }
    HIP_CHECK(hipHostUnregister(A));
  }

  free(A);
  free(Bh);
  HIP_CHECK(hipFree(Bd));
}

template <typename T> __global__ void fill_kernel(T* dataPtr, T value) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  dataPtr[tid] = value;
}

TEMPLATE_TEST_CASE("Unit_hipHostRegister_Flags", "", int, float, double) {
  size_t sizeBytes = 1 * sizeof(TestType);
  TestType* hostPtr = reinterpret_cast<TestType*>(malloc(sizeBytes));

  /* Flags aren't used for AMD devices currently */
  struct FlagType {
    unsigned int value;
    bool valid;
  };

  /* EXSWCPHIPT-29 - 0x08 is hipHostRegisterReadOnly which currently doesn't have a definition in the headers */
  /* hipHostRegisterIoMemory is a valid flag but requires access to I/O mapped memory to be tested */
  FlagType flags = GENERATE(
      FlagType{hipHostRegisterDefault, true}, FlagType{hipHostRegisterPortable, true},
      FlagType{0x08, true}, FlagType{hipHostRegisterPortable | hipHostRegisterMapped, true},
      FlagType{hipHostRegisterPortable | hipHostRegisterMapped | 0x08, true}, FlagType{0xF0, false},
      FlagType{0xFFF2, false}, FlagType{0xFFFFFFFF, false});

  INFO("Testing hipHostRegister flag: " << flags.value);
  if (flags.valid) {
    HIP_CHECK(hipHostRegister(hostPtr, sizeBytes, flags.value));
    HIP_CHECK(hipHostUnregister(hostPtr));
  } else {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, sizeBytes, flags.value), hipErrorInvalidValue);
  }

  free(hostPtr);
}

TEMPLATE_TEST_CASE("Unit_hipHostRegister_Negative", "", int, float, double) {
  TestType* hostPtr = nullptr;

  size_t sizeBytes = 1 * sizeof(TestType);
  SECTION("hipHostRegister Negative Test - nullptr") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, 1, 0), hipErrorInvalidValue);
  }

  hostPtr = reinterpret_cast<TestType*>(malloc(sizeBytes));
  SECTION("hipHostRegister Negative Test - zero size") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, 0, 0), hipErrorInvalidValue);
  }

  size_t devMemAvail{0}, devMemFree{0};
  HIP_CHECK(hipMemGetInfo(&devMemFree, &devMemAvail));
  auto hostMemFree = HipTest::getMemoryAmount() /* In MB */ * 1024 * 1024;  // In bytes
  REQUIRE(devMemFree > 0);
  REQUIRE(devMemAvail > 0);
  REQUIRE(hostMemFree > 0);

  size_t memFree = (std::max)(devMemFree, hostMemFree);  // which is the limiter cpu or gpu

  SECTION("hipHostRegister Negative Test - invalid memory size") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, memFree, 0), hipErrorInvalidValue);
  }

  free(hostPtr);
  SECTION("hipHostRegister Negative Test - freed memory") {
    HIP_CHECK_ERROR(hipHostRegister(hostPtr, 0, 0), hipErrorInvalidValue);
  }
}
