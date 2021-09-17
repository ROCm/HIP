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

/* This testfile verifies the basic functionality of
   hipExtLaunchMultiKernelMultiDevice API.
   It can be tested on single GPU or multi GPUs.
*/


#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include "hip/hip_runtime.h"

#define MAX_GPUS 8
#define NUM_KERNEL_ARGS 3

/*
This testcase verifies hipExtLaunchMultiKernelMultiDevice API for different
datatypes where
1. Intitialize device variables
2. Initializing hipLaunchParams structure to pass it to
   hipExtLaunchMultiKernelMultiDevice API
3. Launches vector_square kernel which performs square of the variable
4. Validates the result with the square of variable.
*/

TEMPLATE_TEST_CASE("Unit_hipExtLaunchMultiKernelMultiDevice_Basic", "", int
                   , float, double) {
  TestType *A_d[MAX_GPUS], *C_d[MAX_GPUS];
  TestType *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(TestType);
  int nGpu = 0;

  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    SUCCEED("info: didn't find any GPU! Skipping the testcase");
  } else {
    if (nGpu > MAX_GPUS) {
      nGpu = MAX_GPUS;
    }
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, nullptr, &C_h, N, false);
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    // Allocating and initializing device variables
    hipStream_t stream[MAX_GPUS];
    for (int i = 0; i < nGpu; i++) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipStreamCreateWithFlags(&stream[i], hipStreamNonBlocking));
      hipDeviceProp_t props;
      HIP_CHECK(hipGetDeviceProperties(&props, i/*deviceID*/));
      INFO("Running on bus 0x" << props.pciBusID << " " << props.name);
      INFO("Allocate device mem " << 2*Nbytes/1024.0/1024.0);
      HIP_CHECK(hipMalloc(&A_d[i], Nbytes));
      HIP_CHECK(hipMalloc(&C_d[i], Nbytes));
      HIP_CHECK(hipMemcpy(A_d[i], A_h, Nbytes, hipMemcpyHostToDevice));
    }

    hipLaunchParams *launchParamsList = reinterpret_cast<hipLaunchParams *>(
        malloc(sizeof(hipLaunchParams)*nGpu));
    void *args[MAX_GPUS * NUM_KERNEL_ARGS];

    // Intializing the hipLaunchParams structure with device variables
    // ,kernel and launching hipExtLaunchMultiKernelMultiDevice API
    for (int i = 0; i < nGpu; i++) {
      args[i * NUM_KERNEL_ARGS]     = &A_d[i];
      args[i * NUM_KERNEL_ARGS + 1] = &C_d[i];
      args[i * NUM_KERNEL_ARGS + 2] = &N;
      launchParamsList[i].func  =
        reinterpret_cast<void *>(HipTest::vector_square<TestType>);
      launchParamsList[i].gridDim   = dim3(blocks);
      launchParamsList[i].blockDim  = dim3(threadsPerBlock);
      launchParamsList[i].sharedMem = 0;
      launchParamsList[i].stream    = stream[i];
      launchParamsList[i].args      = args + i * NUM_KERNEL_ARGS;
    }

    hipExtLaunchMultiKernelMultiDevice(launchParamsList, nGpu, 0);

    // Validating the result
    for (int j = 0; j < nGpu; j++) {
      hipStreamSynchronize(stream[j]);
      hipDeviceProp_t props;
      HIP_CHECK(hipGetDeviceProperties(&props, j/*deviceID*/));
      INFO("Checking result on bus " << props.pciBusID << props.name);

      HIP_CHECK(hipSetDevice(j));
      HIP_CHECK(hipMemcpy(C_h, C_d[j], Nbytes, hipMemcpyDeviceToHost));

      for (size_t i = 0; i < N; i++)  {
        if (C_h[i] != A_h[i] * A_h[i]) {
          INFO("validation failed " << C_h[i] << A_h[i]*A_h[i]);
          REQUIRE(false);
        }
      }
    }

    // DeAllocating memory
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                  A_h, nullptr, C_h, false);
    for (int j = 0; j < nGpu; j++) {
      HIP_CHECK(hipFree(A_d[j]));
      HIP_CHECK(hipFree(C_d[j]));
      HIP_CHECK(hipStreamDestroy(stream[j]));
    }
  }
}
