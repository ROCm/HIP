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
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2
 * TEST: %t --tests 0x3
 * HIT_END
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>
#include "test_common.h"

#define TEST_ITERATIONS 1000
#define CODEOBJ_FILE "kernel_composite_test.code"
/**
 * Run Valgrind tool with these test cases to validate memory leakage.
 * E.g. valgrind --leak-check=yes ./a.out --tests 0x1
 */

/**
 * Internal Function
 */
std::vector<char> load_file() {
  std::ifstream file(CODEOBJ_FILE, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    failed("could not open code object '%s'\n", CODEOBJ_FILE);
  }
  file.close();
  return buffer;
}
/**
 * Validates no memory leakage for hipModuleLoad
 */
void testhipModuleLoadUnloadStress() {
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    hipFunction_t Function;
    HIPCHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIPCHECK(hipModuleUnload(Module));
  }
}
/**
 * Validates no memory leakage for hipModuleLoadData
 */
void testhipModuleLoadDataUnloadStress() {
  auto buffer = load_file();
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIPCHECK(hipModuleLoadData(&Module, &buffer[0]));
    hipFunction_t Function;
    HIPCHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIPCHECK(hipModuleUnload(Module));
  }
}
/**
 * Validates no memory leakage for hipModuleLoadDataEx
 */
void testhipModuleLoadDataExUnloadStress() {
  auto buffer = load_file();
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIPCHECK(hipModuleLoadDataEx(&Module, &buffer[0], 0,
                                nullptr, nullptr));
    hipFunction_t Function;
    HIPCHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIPCHECK(hipModuleUnload(Module));
  }
}

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipCtxCreate(&context, 0, device));
#endif
  if (p_tests == 0x1) {
    testhipModuleLoadUnloadStress();
  } else if (p_tests == 0x2) {
    testhipModuleLoadDataUnloadStress();
  } else if (p_tests == 0x3) {
    testhipModuleLoadDataExUnloadStress();
  }
#ifdef __HIP_PLATFORM_NVCC__
  HIPCHECK(hipCtxDestroy(context));
#endif
  passed();
}
