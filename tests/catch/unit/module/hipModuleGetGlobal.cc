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


#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <fstream>
#include <vector>

#define LEN 64
#define SIZE LEN * sizeof(float)
#define ARRAY_SIZE 16
#define fileName "module_kernels.code"

/*
This testcase verifies the basic functionality of hipModuleGetGlobal API
1. Simple kernel
2. Global variables
*/
TEST_CASE("Unit_hipModuleGetGlobal_Basic") {
  float *A{nullptr}, *B{nullptr}, *Ad{nullptr}, *Bd{nullptr};
  HipTest::initArrays<float>(&Ad, &Bd, nullptr, &A, &B, nullptr, LEN,
                             false);
  CTX_CREATE()
  hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(Ad), A, SIZE);
  hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(Bd), B, SIZE);
  hipModule_t Module;
  HIP_CHECK(hipModuleLoad(&Module, fileName));

  float myDeviceGlobal_h = 42.0;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize,
                               Module, "myDeviceGlobal"));
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(deviceGlobal),
                          &myDeviceGlobal_h, deviceGlobalSize));
  float myDeviceGlobalArray_h[ARRAY_SIZE];
  hipDeviceptr_t myDeviceGlobalArray;
  size_t myDeviceGlobalArraySize;

  HIP_CHECK(hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>
                               (&myDeviceGlobalArray),
                               &myDeviceGlobalArraySize, Module,
                               "myDeviceGlobalArray"));

  for (int i = 0; i < ARRAY_SIZE; i++) {
    myDeviceGlobalArray_h[i] = i * 1000.0f;
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>
                            (myDeviceGlobalArray),
                            &myDeviceGlobalArray_h,
                            myDeviceGlobalArraySize));
  }

  struct {
    void* _Ad;
    void* _Bd;
  } args;

  args._Ad = reinterpret_cast<void*>(Ad);
  args._Bd = reinterpret_cast<void*>(Bd);
  size_t size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  SECTION("Testing with simple kernel") {
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "hello_world"));
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0,
                                    NULL,
                                    reinterpret_cast<void**>(&config)));

    hipMemcpyDtoH(B, hipDeviceptr_t(Bd), SIZE);

    for (uint32_t i = 0; i < LEN; i++) {
      REQUIRE(A[i] == B[i]);
    }
  }

  SECTION("Testing global variables") {
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "test_globals"));
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0,
                                    NULL,
                                    reinterpret_cast<void**>(&config)));

    hipMemcpyDtoH(B, hipDeviceptr_t(Bd), SIZE);

    for (uint32_t i = 0; i < LEN; i++) {
      float expected = A[i] + myDeviceGlobal_h +
        myDeviceGlobalArray_h[i % 16];
      REQUIRE(expected == B[i]);
    }
  }

  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
  HipTest::freeArrays<float>(Ad, Bd, nullptr,
                             A, B, nullptr,
                             false);
}
