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
hipManagedKeyword API Scenario
1. Test hipModuleLoad on multiple GPUs
*/

#include "hip_test_common.hh"
#include "hip_test_kernels.hh"
#include "hip_test_checkers.hh"

#define MANAGED_VAR_INIT_VALUE 10
#define fileName "module_kernels.code"

TEST_CASE("Unit_hipMangedKeyword_ModuleLoadMultiGPU") {
  int numDevices = 0, data;
  hipDeviceptr_t x;
  size_t xSize;
  hipGetDeviceCount(&numDevices);
  for (int i = 0; i < numDevices; i++) {
    hipSetDevice(i);
    CTX_CREATE()
    hipModule_t Module;
    HIP_CHECK(hipModuleLoad(&Module, fileName));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "GPU_func"));
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, 1, 1,
                                    1, 0, 0, NULL, NULL));
    hipDeviceSynchronize();
    HIP_CHECK(hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(&x),
                                 &xSize, Module, "x"));
    HIP_CHECK(hipMemcpyDtoH(&data, hipDeviceptr_t(x), xSize));
    REQUIRE(data == (1 + MANAGED_VAR_INIT_VALUE));
    HIP_CHECK(hipModuleUnload(Module));
    CTX_DESTROY()
  }
}
