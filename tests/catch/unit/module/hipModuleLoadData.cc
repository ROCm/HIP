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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/*
hipModuleLoadData scenarios

1. Loads the kernel and the corresponding kernel function
   which copies the data from one device variable to another.
*/

#include <fstream>
#include <vector>
#include "hip_test_common.hh"
#include "hip_test_checkers.hh"

#define LEN 64
#define SIZE LEN << 2
#define FILENAME "module_kernels.code"
#define kernel_name "hello_world"

static std::vector<char> load_file() {
  std::ifstream file(FILENAME, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    INFO("could not open code object" << FILENAME);
    REQUIRE(false);
  }
  return buffer;
}


TEST_CASE("Unit_hipModuleLoadData_Basic") {
    auto buffer = load_file();
    float *A{nullptr}, *B{nullptr}, *Ad{nullptr}, *Bd{nullptr};
    HipTest::initArrays<float>(&Ad, &Bd, nullptr, &A, &B, nullptr,
                               LEN, false);
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

    hipModule_t Module;
    hipFunction_t Function{nullptr};

    HIP_CHECK(hipModuleLoadData(&Module, &buffer[0]));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

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
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0,
          stream, NULL, reinterpret_cast<void**>(&config)));

    HIP_CHECK(hipStreamDestroy(stream));

    HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));

    for (uint32_t i = 0; i < LEN; i++) {
      REQUIRE(A[i] == B[i]);
    }
    HipTest::freeArrays<float>(Ad, Bd, nullptr,
                        A, B,
                        nullptr, false);
}
