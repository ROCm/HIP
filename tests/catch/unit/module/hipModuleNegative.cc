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
This testcase verifies the negative scenarios of
1. hipModuleLoad API
2. hipModuleLoadData API
3. hipModuleGetFunction API
4. hipModuleGetGlobal API
*/

#include <ctime>
#include <fstream>
#include <cstddef>
#include <vector>
#include "hip_test_common.hh"

#define FILENAME_NONEXST "sample_nonexst.code"
#define FILENAME_EMPTY "emptyfile.code"
#define FILENAME_RAND "rand_file.code"
#define RANDOMFILE_LEN 2048
#define CODEOBJ_FILE "module_kernels.code"
#define KERNEL_NAME "hello_world"
#define KERNEL_NAME_NONEXST "xyz"
#define CODEOBJ_GLOBAL "module_kernels.code"
#define DEVGLOB_VAR_NONEXIST "xyz"
#define DEVGLOB_VAR "myDeviceGlobal"
/**
 * Internal Function
 * Loads the kernel file into buffer
 */
std::vector<char> load_file(const char* filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    INFO("could not open code object " <<  filename);
  }
  file.close();
  return buffer;
}

/**
 * Internal Function
   Create Randome file
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
 * Validates negative scenarios for hipModuleLoad API
 */

TEST_CASE("Unit_hipModuleLoad_Negative") {
  CTX_CREATE()
  hipModule_t Module;

  SECTION("Nullptr to module") {
    REQUIRE(hipModuleLoad(nullptr, CODEOBJ_FILE)
            != hipSuccess);
  }

  SECTION("Nullptr to Fname") {
    REQUIRE(hipModuleLoad(&Module, nullptr)
            != hipSuccess);
  }

  SECTION("Empty fname") {
  std::fstream fs;
  fs.open(FILENAME_EMPTY, std::ios::out);
  fs.close();
  REQUIRE(hipModuleLoad(&Module, FILENAME_EMPTY)
                        != hipSuccess);
  }

  SECTION("Binary file with random number") {
    createRandomFile(FILENAME_RAND);
    REQUIRE(hipModuleLoad(&Module, FILENAME_RAND)
                          != hipSuccess);
    remove(FILENAME_RAND);
  }

  SECTION("Non Existent file") {
    REQUIRE(hipModuleLoad(&Module, FILENAME_NONEXST)
                          != hipSuccess);
  }

  SECTION("Empty string to file name") {
    REQUIRE(hipModuleLoad(&Module, "")
                          != hipSuccess);
  }

  CTX_DESTROY()
}

/**
 * Validates negative scenarios for hipModuleLoadData API
 */
TEST_CASE("Unit_hipModuleLoadData_Negative") {
  CTX_CREATE()
  hipModule_t Module;

  SECTION("Nullptr to module") {
    auto buffer = load_file(CODEOBJ_FILE);
    REQUIRE(hipModuleLoadData(nullptr, &buffer[0])
                              != hipSuccess);
  }

  SECTION("Nullptr to image") {
    REQUIRE(hipModuleLoadData(&Module, nullptr)
                              != hipSuccess);
  }

  SECTION("Random file to image") {
    createRandomFile(FILENAME_RAND);
    auto buffer = load_file(FILENAME_RAND);
    REQUIRE(hipModuleLoadData(&Module, &buffer[0])
                              != hipSuccess);
  }

  SECTION("Nullptr to Module") {
  auto buffer = load_file(CODEOBJ_FILE);
  REQUIRE(hipModuleLoadDataEx(nullptr, &buffer[0], 0, nullptr, nullptr)
                              != hipSuccess);
  }

  SECTION("Nullptr to image") {
    REQUIRE(hipModuleLoadDataEx(&Module, nullptr, 0, nullptr, nullptr)
                                != hipSuccess);
  }

  SECTION("Random image file") {
    // Create a binary file with random numbers
    createRandomFile(FILENAME_RAND);
    // Open the code object file and copy it in a buffer
    auto buffer = load_file(FILENAME_RAND);
    REQUIRE(hipModuleLoadDataEx(&Module, &buffer[0], 0, nullptr, nullptr)
        != hipSuccess);
  }

  CTX_DESTROY()
}

/**
 * Validates negative scenarios for hipModuleGetFunction API
 */
TEST_CASE("Unit_hipModuleGetFunction_Negative") {
  CTX_CREATE()
  hipFunction_t Function;
  hipModule_t Module;

  SECTION("Nullptr to function name") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    REQUIRE(hipModuleGetFunction(nullptr, Module, KERNEL_NAME) != hipSuccess);
    HIP_CHECK(hipModuleUnload(Module));
  }


  SECTION("Non existing function kernel name") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    REQUIRE(hipModuleGetFunction(&Function, Module, KERNEL_NAME_NONEXST)
                                 != hipSuccess);
    HIP_CHECK(hipModuleUnload(Module));
  }

  SECTION("Nullptr to kernel name") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    REQUIRE(hipModuleGetFunction(&Function, Module, nullptr) != hipSuccess);
    HIP_CHECK(hipModuleUnload(Module));
  }
#if HT_AMD
  SECTION("Uninitialized module") {
    REQUIRE(hipModuleGetFunction(&Function, Module, KERNEL_NAME) != hipSuccess);
  }

  SECTION("Unloaded module") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    HIP_CHECK(hipModuleUnload(Module));
    REQUIRE(hipModuleGetFunction(&Function, Module, KERNEL_NAME) != hipSuccess);
  }
#endif

  SECTION("Empty string to kernel name") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    REQUIRE(hipModuleGetFunction(&Function, Module, "") != hipSuccess);
    HIP_CHECK(hipModuleUnload(Module));
  }

  CTX_DESTROY()
}

/**
 * Validates negative scenarios for hipModuleGetGlobal API
 */
TEST_CASE("Unit_hipModuleGetGlobal_Negative") {
  CTX_CREATE()
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;

  SECTION("Nullptr to varname") {
    HIPCHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
    REQUIRE(hipModuleGetGlobal(&deviceGlobal,
                               &deviceGlobalSize, Module, nullptr)
                               != hipSuccess);
    HIPCHECK(hipModuleUnload(Module));
  }

  SECTION("Wrong variable name") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
    REQUIRE(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize,
                               Module, DEVGLOB_VAR_NONEXIST) != hipSuccess);
    HIPCHECK(hipModuleUnload(Module));
  }

  SECTION("Empty string to module name") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
    REQUIRE(hipModuleGetGlobal(&deviceGlobal,
                               &deviceGlobalSize, Module, "") != hipSuccess);
    HIPCHECK(hipModuleUnload(Module));
  }

#if HT_AMD
  SECTION("Unloaded Module") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
    HIP_CHECK(hipModuleUnload(Module));
    REQUIRE(hipModuleGetGlobal(&deviceGlobal,
                               &deviceGlobalSize, Module,
                               DEVGLOB_VAR) != hipSuccess);
  }

  SECTION("Unload an Unloaded module") {
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    HIP_CHECK(hipModuleUnload(Module));
    REQUIRE(hipModuleUnload(Module) != hipSuccess);
  }

  SECTION("Uninitialized module") {
    REQUIRE(hipModuleGetGlobal(&deviceGlobal,
                               &deviceGlobalSize, Module,
                               DEVGLOB_VAR) != hipSuccess);
  }
  SECTION("Unload Uninitialized module") {
    REQUIRE(hipModuleUnload(Module) != hipSuccess);
  }
#endif

  CTX_DESTROY()
}
