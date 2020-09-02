/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x10 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x11 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x12 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x13
 * TEST: %t --tests 0x14
 * TEST: %t --tests 0x15
 * TEST: %t --tests 0x20 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x21 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x22
 * TEST: %t --tests 0x30 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x31 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x32
 * TEST: %t --tests 0x40 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x41
 * TEST: %t --tests 0x42
 * TEST: %t --tests 0x43 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x44 EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t --tests 0x45
 * TEST: %t --tests 0x50 EXCLUDE_HIP_PLATFORM hcc rocclr nvcc
 * TEST: %t --tests 0x51 EXCLUDE_HIP_PLATFORM hcc rocclr nvcc
 * TEST: %t --tests 0x52 EXCLUDE_HIP_PLATFORM hcc rocclr
 * TEST: %t --tests 0x53
 * TEST: %t --tests 0x54
 * TEST: %t --tests 0x55 EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t --tests 0x56
 * TEST: %t --tests 0x60 EXCLUDE_HIP_PLATFORM nvcc
 * HIT_END
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>
#include "test_common.h"

#define FILENAME_NONEXST "sample_nonexst.code"
#define FILENAME_EMPTY "emptyfile.code"
#define FILENAME_RAND "rand_file.code"
#define RANDOMFILE_LEN 2048
#define CODEOBJ_FILE "vcpy_kernel.code"
#define KERNEL_NAME "hello_world"
#define KERNEL_NAME_NONEXST "xyz"
#define CODEOBJ_GLOBAL "global_kernel.code"
#define DEVGLOB_VAR_NONEXIST "xyz"
#define DEVGLOB_VAR "myDeviceGlobal"
/**
 * Internal Function
 */
std::vector<char> load_file(const char* filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    failed("could not open code object '%s'\n", filename);
  }
  file.close();
  return buffer;
}

/**
 * Internal Function
 */
void createRandomFile(const char* filename) {
  std::ofstream outfile(filename, std::ios::binary);
  char buf[RANDOMFILE_LEN];
  unsigned int seed = 1;
  for (int i = 0; i < RANDOMFILE_LEN; i++) {
    buf[i] = rand_r(&seed) % 256;
  }
  outfile.write(buf, RANDOMFILE_LEN);
  outfile.close();
}

/**
 * Internal Function
 */
#ifdef __HIP_PLATFORM_NVCC__
void initHipCtx(hipCtx_t *pcontext) {
  HIPCHECK(hipInit(0));
  hipDevice_t device;
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipCtxCreate(pcontext, 0, device));
}
#endif

/**
 * Validates negative scenarios for hipModuleLoad
 * module = nullptr
 */
bool testhipModuleLoadNeg10() {
  bool TestPassed = false;
  hipError_t ret;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoad(nullptr, CODEOBJ_FILE))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = nullptr
 */
bool testhipModuleLoadNeg11() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoad(&Module, nullptr))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}
/**
 * Validates negative scenarios for hipModuleLoad
 * fname = empty file
 */
bool testhipModuleLoadNeg12() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create an empty
  std::fstream fs;
  fs.open(FILENAME_EMPTY, std::ios::out);
  fs.close();
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoad(&Module, FILENAME_EMPTY))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_EMPTY);
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = ramdom file
 */
bool testhipModuleLoadNeg13() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create a binary file with random numbers
  createRandomFile(FILENAME_RAND);
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoad(&Module, FILENAME_RAND))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_RAND);
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = non existent file
 */
bool testhipModuleLoadNeg14() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoad(&Module, FILENAME_NONEXST))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = empty string ""
 */
bool testhipModuleLoadNeg15() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoad(&Module, "")) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadData
 * module = nullptr
 */
bool testhipModuleLoadDataNeg20() {
  bool TestPassed = false;
  hipError_t ret;
  auto buffer = load_file(CODEOBJ_FILE);
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoadData(nullptr, &buffer[0]))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadData
 * image = nullptr
 */
bool testhipModuleLoadDataNeg21() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoadData(&Module, nullptr))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadData
 * image = ramdom file
 */
bool testhipModuleLoadDataNeg22() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create a binary file with random numbers
  createRandomFile(FILENAME_RAND);
  // Open the code object file and copy it in a buffer
  auto buffer = load_file(FILENAME_RAND);
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoadData(&Module, &buffer[0]))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_RAND);
  return TestPassed;
}
/**
 * Validates negative scenarios for hipModuleLoadDataEx
 * module = nullptr
 */
bool testhipModuleLoadDataExNeg30() {
  bool TestPassed = false;
  hipError_t ret;
  // Open the code object file and copy it in a buffer
  auto buffer = load_file(CODEOBJ_FILE);
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoadDataEx(nullptr, &buffer[0], 0, nullptr, nullptr))
       != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadDataEx
 * image = nullptr
 */
bool testhipModuleLoadDataExNeg31() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoadDataEx(&Module, nullptr, 0, nullptr, nullptr))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadDataEx
 * image = ramdom file
 */
bool testhipModuleLoadDataExNeg32() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create a binary file with random numbers
  createRandomFile(FILENAME_RAND);
  // Open the code object file and copy it in a buffer
  auto buffer = load_file(FILENAME_RAND);
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleLoadDataEx(&Module, &buffer[0], 0, nullptr, nullptr))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_RAND);
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * Function = nullptr
 */
bool testhipModuleGetFunctionNeg40() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if ((ret = hipModuleGetFunction(nullptr, Module, KERNEL_NAME))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * Module is uninitialized
 */
bool testhipModuleGetFunctionNeg41() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleGetFunction(&Function, Module, KERNEL_NAME))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * kname = non existing function
 */
bool testhipModuleGetFunctionNeg42() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if ((ret = hipModuleGetFunction(&Function, Module, KERNEL_NAME_NONEXST))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * kname = nullptr
 */
bool testhipModuleGetFunctionNeg43() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if ((ret = hipModuleGetFunction(&Function, Module, nullptr))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * Module = Unloaded Module
 */
bool testhipModuleGetFunctionNeg44() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIPCHECK(hipModuleUnload(Module));
  if ((ret = hipModuleGetFunction(&Function, Module, KERNEL_NAME))
      != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * kname = Empty String ""
 */
bool testhipModuleGetFunctionNeg45() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if ((ret = hipModuleGetFunction(&Function,
                       Module, "")) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * dptr = nullptr
 */
bool testhipModuleGetGlobalNeg50() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  size_t deviceGlobalSize;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  if ((ret = hipModuleGetGlobal(nullptr,
    &deviceGlobalSize, Module, DEVGLOB_VAR)) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * bytes = nullptr
 */
bool testhipModuleGetGlobalNeg51() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  if ((ret = hipModuleGetGlobal(&deviceGlobal, nullptr,
      Module, DEVGLOB_VAR)) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * name = nullptr
 */
bool testhipModuleGetGlobalNeg52() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  if ((ret = hipModuleGetGlobal(&deviceGlobal,
    &deviceGlobalSize, Module, nullptr)) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * name = wrong name
 */
bool testhipModuleGetGlobalNeg53() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  if ((ret = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize,
    Module, DEVGLOB_VAR_NONEXIST)) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * name = Empty String ""
 */
bool testhipModuleGetGlobalNeg54() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  if ((ret = hipModuleGetGlobal(&deviceGlobal,
    &deviceGlobalSize, Module, "")) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
  HIPCHECK(hipModuleUnload(Module));
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * Module = Unloaded Module
 */
bool testhipModuleGetGlobalNeg55() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  HIPCHECK(hipModuleUnload(Module));
  if ((ret = hipModuleGetGlobal(&deviceGlobal,
    &deviceGlobalSize, Module, DEVGLOB_VAR)) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * Module = Uninitialized Module
 */
bool testhipModuleGetGlobalNeg56() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  if ((ret = hipModuleGetGlobal(&deviceGlobal,
    &deviceGlobalSize, Module, DEVGLOB_VAR)) != hipSuccess) {
    TestPassed = true;
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleUnload
 * 1. Unload an uninitialized module
 * 2. Unload an unloaded module
 */
bool testhipModuleLoadNeg60() {
  bool TestPassed = true;
  hipError_t ret;
  hipModule_t Module;
#ifdef __HIP_PLATFORM_NVCC__
  hipCtx_t context;
  initHipCtx(&context);
#endif
  // test case 1
  if ((ret = hipModuleUnload(Module)) != hipSuccess) {
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  } else {
    TestPassed &= false;
  }
  // test case 2
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIPCHECK(hipModuleUnload(Module));
  if ((ret = hipModuleUnload(Module)) != hipSuccess) {
    printf("Test Passed: Error Code Returned: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
  } else {
    TestPassed &= false;
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;
  if (p_tests == 0x10) {
    TestPassed = testhipModuleLoadNeg10();
  } else if (p_tests == 0x11) {
    TestPassed = testhipModuleLoadNeg11();
  } else if (p_tests == 0x12) {
    TestPassed = testhipModuleLoadNeg12();
  } else if (p_tests == 0x13) {
    TestPassed = testhipModuleLoadNeg13();
  } else if (p_tests == 0x14) {
    TestPassed = testhipModuleLoadNeg14();
  } else if (p_tests == 0x15) {
    TestPassed = testhipModuleLoadNeg15();
  } else if (p_tests == 0x20) {
    TestPassed = testhipModuleLoadDataNeg20();
  } else if (p_tests == 0x21) {
    TestPassed = testhipModuleLoadDataNeg21();
  } else if (p_tests == 0x22) {
    TestPassed = testhipModuleLoadDataNeg22();
  } else if (p_tests == 0x30) {
    TestPassed = testhipModuleLoadDataExNeg30();
  } else if (p_tests == 0x31) {
    TestPassed = testhipModuleLoadDataExNeg31();
  } else if (p_tests == 0x32) {
    TestPassed = testhipModuleLoadDataExNeg32();
  } else if (p_tests == 0x40) {
    TestPassed = testhipModuleGetFunctionNeg40();
  } else if (p_tests == 0x41) {
    TestPassed = testhipModuleGetFunctionNeg41();
  } else if (p_tests == 0x42) {
    TestPassed = testhipModuleGetFunctionNeg42();
  } else if (p_tests == 0x43) {
    TestPassed = testhipModuleGetFunctionNeg43();
  } else if (p_tests == 0x44) {
    TestPassed = testhipModuleGetFunctionNeg44();
  } else if (p_tests == 0x45) {
    TestPassed = testhipModuleGetFunctionNeg45();
  } else if (p_tests == 0x50) {
    TestPassed = testhipModuleGetGlobalNeg50();
  } else if (p_tests == 0x51) {
    TestPassed = testhipModuleGetGlobalNeg51();
  } else if (p_tests == 0x52) {
    TestPassed = testhipModuleGetGlobalNeg52();
  } else if (p_tests == 0x53) {
    TestPassed = testhipModuleGetGlobalNeg53();
  } else if (p_tests == 0x54) {
    TestPassed = testhipModuleGetGlobalNeg54();
  } else if (p_tests == 0x55) {
    TestPassed = testhipModuleGetGlobalNeg55();
  } else if (p_tests == 0x56) {
    TestPassed = testhipModuleGetGlobalNeg56();
  } else if (p_tests == 0x60) {
    TestPassed = testhipModuleLoadNeg60();
  } else {
    printf("Invalid Test Case \n");
    exit(1);
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
}
