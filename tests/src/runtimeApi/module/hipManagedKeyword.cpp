/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD_CMD: managed_kernel.code %hc --genco %S/managed_kernel.cpp -o managed_kernel.code EXCLUDE_HIP_PLATFORM amd
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM amd
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include <iostream>
#include "test_common.h"

#define MANAGED_VAR_INIT_VALUE 10
#define fileName "managed_kernel.code"

bool managedMultiGPUTest() {
  int numDevices = 0;
  hipDeviceptr_t x;
  size_t xSize;
  int data;
  hipGetDeviceCount(&numDevices);
  for (int i = 0; i < numDevices; i++) {
    hipSetDevice(i);
    hipDevice_t device;
    hipCtx_t context;
    hipDeviceGet(&device, i);
    hipCtxCreate(&context, 0, device);
    hipModule_t Module;
    HIPCHECK(hipModuleLoad(&Module, fileName));
    hipFunction_t Function;
    HIPCHECK(hipModuleGetFunction(&Function, Module, "GPU_func"));
    HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, 1, 1, 1, 0, 0, NULL, NULL));
    hipDeviceSynchronize();
    HIPCHECK(hipModuleGetGlobal((hipDeviceptr_t*)&x, &xSize, Module, "x"));
    HIPCHECK(hipMemcpyDtoH(&data, hipDeviceptr_t(x), xSize));
    if (data != (1 + MANAGED_VAR_INIT_VALUE)) {
      HIPCHECK(hipModuleUnload(Module));
      hipCtxDestroy(context);
      return false;
    }
    HIPCHECK(hipModuleUnload(Module));
    hipCtxDestroy(context);
  }
  return true;
}

int main(int argc, char** argv) {
  hipInit(0);
  bool testStatus = managedMultiGPUTest();
  if (!testStatus) {
    failed("Managed keyword module test failed!");
  }
  passed();
}
