/*
Copyright (c) 2017-present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD_CMD: global_kernel.code %hc --genco %S/global_kernel.cpp -o global_kernel.code
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <hip/hip_hcc.h>

#include "test_common.h"

#define dynFileName1 "dyn_global_kernel1.code"
#define dynFileName2 "dyn_global_kernel2.code"

__device__ char statSingleVar;
__device__ char statDynGlobal;
__device__ char statTexGlobal;
__device__ char statDynTexGlobal;

__device__ int shadowGlobal;
texture<float, 2, hipReadModeElementType> texGlobal;

#define LEN 64
__device__ int statSymcopyGlobal[LEN];

#define HIP_CHECK(cmd)                                                                             \
    {                                                                                              \
        hipError_t status = cmd;                                                                   \
        if (status != hipSuccess) {                                                                \
            std::cout << "error: #" << status << " (" << hipGetErrorString(status)                 \
                      << ") at line:" << __LINE__ << ":  " << #cmd << std::endl;                   \
            abort();                                                                               \
        }                                                                                          \
    }


bool basic_check() {
  /*! Init Vars */
  hipModule_t Module1;
  hipModule_t Module2;

  hipError_t error = hipSuccess;
  hipDeviceptr_t deviceGlobal = nullptr;
  size_t deviceGlobalSize = 0;

  /*! Load Modules */
  HIP_CHECK(hipModuleLoad(&Module1, dynFileName1));
  HIP_CHECK(hipModuleLoad(&Module2, dynFileName2));

  /*! Case 0: Try Var Not present*/
  error = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, nullptr, "WrongGlobalVar");
  assert(error == hipErrorNotFound);

  error = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module1, "WrongGlobalVar");
  assert(error == hipErrorNotFound);

  error = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module2, "WrongGlobalVar");
  assert(error == hipErrorNotFound);


  /*! Case 1: Single Var present - basic check*/
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, nullptr, "statSingleVar"));
  assert(deviceGlobalSize == sizeof(char));

  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module1, "dynSingleVar"));
  assert(deviceGlobalSize == sizeof(float));

  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module2, "texSingleVar"));
  assert(deviceGlobalSize == 0);

  /*! Case 2: Single Var present - var without modules, queried with wrong module */
  error = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module1, "statSingleVar");
  assert(error == hipErrorNotFound);

  /*! Case 3: Single Var present - var with modules, queried with nullptr */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, nullptr, "dynSingleVar"));
  /* If its a single var, allow the available var to be returned */
  assert(deviceGlobalSize == sizeof(float));

  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, nullptr, "texSingleVar"));
  /* If its a single var, allow the available var to be returned */
  assert(deviceGlobalSize == 0);

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module1));
  HIP_CHECK(hipModuleUnload(Module2));

  return true;
}

bool dyng_duplicates() {

  /*! Init Vars */
  hipModule_t Module1;
  hipModule_t Module2;

  hipError_t error = hipSuccess;
  hipDeviceptr_t deviceGlobal = nullptr;
  size_t deviceGlobalSize = 0;

  /*! Load Modules */
  HIP_CHECK(hipModuleLoad(&Module1, dynFileName1));
  HIP_CHECK(hipModuleLoad(&Module2, dynFileName2));

  /*! Dynamic __device__ duplicate Global Vars */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module1, "dynGlobal"));
  assert(deviceGlobalSize == sizeof(char));

  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module2, "dynGlobal"));
  assert(deviceGlobalSize == sizeof(float));

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module1));
  HIP_CHECK(hipModuleUnload(Module2));

  return true;
}

bool texg_duplicates() {
    /*! Init Vars */
  hipModule_t Module1;
  hipModule_t Module2;

  hipError_t error = hipSuccess;
  hipDeviceptr_t deviceGlobal1 = nullptr;
  hipDeviceptr_t deviceGlobal2 = nullptr;
  size_t deviceGlobalSize = 0;

  /*! Load Modules */
  HIP_CHECK(hipModuleLoad(&Module1, dynFileName1));
  HIP_CHECK(hipModuleLoad(&Module2, dynFileName2));

  /*! __hip_pinned_shadow__ duplicate Global Vars */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal1, &deviceGlobalSize, Module1, "texGlobal"));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal2, &deviceGlobalSize, Module2, "texGlobal"));

  /*! Assert that the address is different between 2 duplicate tex vars */
  assert(deviceGlobal1 != deviceGlobal2);

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module1));
  HIP_CHECK(hipModuleUnload(Module2));

  return true;
}

bool statg_dyng_duplicates() {
  hipModule_t Module = nullptr;
  hipError_t error = hipSuccess;
  hipDeviceptr_t deviceGlobal = nullptr;
  size_t deviceGlobalSize = 0;

  HIP_CHECK(hipModuleLoad(&Module, dynFileName1));

  /*! Duplicate __device__ var present in static and dynamic module */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "statDynGlobal"));
  assert(deviceGlobalSize == sizeof(float));

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module));

  return true;
}

bool statg_texg_duplicates() {
  hipModule_t Module = nullptr;
  hipError_t error = hipSuccess;
  hipDeviceptr_t deviceGlobal = nullptr;
  size_t deviceGlobalSize = 0;

  /*! Load Modules */
  HIP_CHECK(hipModuleLoad(&Module, dynFileName2));

  /*! Duplicate __hip_pinned_shadow__ var present in dynamic module and __device__ in static module */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "statTexGlobal"));
  assert(deviceGlobalSize == 0);

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module));

  return true;
}

bool statg_dyng_texg_duplicates() {
  hipModule_t Module1;
  hipModule_t Module2;

  hipError_t error = hipSuccess;
  hipDeviceptr_t deviceGlobal = nullptr;
  hipDeviceptr_t texGlobal = nullptr;
  size_t deviceGlobalSize = 0;

  /*! Load Modules */
  HIP_CHECK(hipModuleLoad(&Module1, dynFileName1));
  HIP_CHECK(hipModuleLoad(&Module2, dynFileName2));

  /*! Duplicate __hip_pinned_shadow__ var present in dynamic module and __device__ in static module */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module1, "statDynTexGlobal"));
  assert(deviceGlobalSize == sizeof(float));

  HIP_CHECK(hipModuleGetGlobal(&texGlobal, &deviceGlobalSize, Module2, "statDynTexGlobal"));
  assert(deviceGlobal != texGlobal);

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module1));
  HIP_CHECK(hipModuleUnload(Module2));

  return true;
}

bool nomodule_duplicates() {
  hipModule_t Module1;

  hipError_t error = hipSuccess;
  hipDeviceptr_t deviceGlobal = nullptr;
  size_t deviceGlobalSize = 0;

  /*! Load Modules */
  HIP_CHECK(hipModuleLoad(&Module1, dynFileName1));

  /*! 2 Global Var of same name, when module is not passed for clarity, fail */
  error = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, nullptr, "statDynGlobal");
  assert(error == hipErrorNotFound);

  /*! 1 Global Var, > 1 Shadow Var - pick up the right module */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, nullptr, "shadowGlobal"));
  assert(deviceGlobalSize == sizeof(int));

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module1));

  return true;
}

bool symcopy_duplicates() {
  hipModule_t Module;
  int *Ah1, *Ah2;
  int *Ad1, *Ad2;
  size_t SIZE = (sizeof(int) * LEN);

  hipDeviceptr_t deviceGlobal1 = nullptr;
  hipDeviceptr_t deviceGlobal2 = nullptr;
  size_t deviceGlobalSize = 0;

  /*! Load Modules */
  HIP_CHECK(hipModuleLoad(&Module, dynFileName1));

  Ah1 = new int[LEN];
  Ah2 = new int[LEN];

  for (int idx=0; idx<LEN; ++idx) {
    Ah1[idx] = -1 * idx;
    Ah2[idx] = 0;
  }

  /*! Retrieve Device Pointers for Symcopy vars */
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal1, &deviceGlobalSize, nullptr, "statSymcopyGlobal"));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal2, &deviceGlobalSize, Module, "dynSymcopyGlobal"));

  /*! Copy vars from HostPtr1 --> SymcopyVar1 --[DeviceCopy]--> SymcopyVar2 --> HostPtr2*/
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(statSymcopyGlobal), Ah1, SIZE, 0, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpyDtoD(deviceGlobal2, deviceGlobal1, SIZE));
  HIP_CHECK(hipMemcpyFromSymbol(Ah2, HIP_SYMBOL(dynSymcopyGlobal), SIZE, 0, hipMemcpyDeviceToHost));

  /*! Assert the values match */
  for (int idx=0; idx<LEN; ++idx) {
    assert(Ah1[idx] == Ah2[idx]);
  }

  /*! Unload Modules */
  HIP_CHECK(hipModuleUnload(Module));

  return true;
}

int main() {
  bool testResult = true;
  hipInit(0);

  /*! Basic Checks */
  testResult |= basic_check();

  /*! Same type of Module, duplicate vars */
  testResult |= dyng_duplicates();
  testResult |= texg_duplicates();

  /*! Different type of Modules, duplicate vars */
  testResult |= statg_dyng_duplicates();
  testResult |= statg_texg_duplicates();

  /*! Duplicate var in all 3 modules */
  testResult |= statg_dyng_texg_duplicates();

  /*! When no module is passed */
  testResult |= nomodule_duplicates();

  /*! Test symbol copy APIs */
  testResult |= symcopy_duplicates();

  if (testResult) {
    passed();
  } else {
    failed("Failed Duplicate Symbols test");
  }
  return 0;
}
