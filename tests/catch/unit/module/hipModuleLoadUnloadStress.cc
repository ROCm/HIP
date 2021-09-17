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

#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>
#include "hip_test_common.hh"

#define TEST_ITERATIONS 1000
#define CODEOBJ_FILE "module_kernels.code"
/**
 * Run Valgrind tool with these test cases to validate memory leakage.
 * E.g. valgrind --leak-check=yes ./a.out
 */

/**
 * Internal Function
 */
static std::vector<char> load_file() {
  std::ifstream file(CODEOBJ_FILE, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    WARN("could not open code object " << CODEOBJ_FILE);
  }
  file.close();
  return buffer;
}
/**
 * Validates no memory leakage for hipModuleLoad
 */
TEST_CASE("Unit_hipModule_LoadUnloadStress") {
  CTX_CREATE()
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIP_CHECK(hipModuleUnload(Module));
  }
  CTX_DESTROY()
}
/**
 * Validates no memory leakage for hipModuleLoadData
 */
TEST_CASE("Unit_hipModuleLoadData_LoadUnloadStress") {
  CTX_CREATE()
  auto buffer = load_file();
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIP_CHECK(hipModuleLoadData(&Module, &buffer[0]));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIP_CHECK(hipModuleUnload(Module));
  }
  CTX_DESTROY()
}
/**
 * Validates no memory leakage for hipModuleLoadDataEx
 */
TEST_CASE("Unit_hipModuleLoadDataEx_UnloadStress") {
  CTX_CREATE()
  auto buffer = load_file();
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIP_CHECK(hipModuleLoadDataEx(&Module, &buffer[0], 0,
                                nullptr, nullptr));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIP_CHECK(hipModuleUnload(Module));
  }
  CTX_DESTROY()
}
