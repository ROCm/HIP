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
This testcase verifies the multithreaded scenario of
hipModuleLoadData API on MultiGPU system
*/

#include <fstream>
#include <vector>

#include "hip_test_common.hh"
#include "hip_test_checkers.hh"

#define LEN 64
#define SIZE LEN << 2
#define THREADS 8

#define FILENAME "module_kernels.code"
#define kernel_name "hello_world"

/*
This function reads the kernel code object file into buffer
*/
static std::vector<char> load_file() {
  std::ifstream file(FILENAME, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    INFO("could not open code object " << FILENAME);
    REQUIRE(false);
  }
  return buffer;
}

/*
Thread function
1. Loads the module using hipModuleLoadData API
2. Initializes 2 device variables.
3. Launches kernel which copies one data into another.
4. validates the result and returns it to the caller using
   std::ref variable.
*/
static void run(const std::vector<char>& buffer, int deviceNo,
                bool &testResult) {
  hipSetDevice(deviceNo);
  hipModule_t Module;
  hipFunction_t Function;
  float *A{nullptr}, *B{nullptr}, *Ad{nullptr}, *Bd{nullptr};
  testResult = true;
  HipTest::initArrays<float>(&Ad, &Bd, nullptr,
                             &A, &B, nullptr,
                             LEN, false);
  HIPCHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

  HIPCHECK(hipModuleLoadData(&Module, &buffer[0]));
  HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = static_cast<void*>(Ad);
  args._Bd = static_cast<void*>(Bd);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN,
                                 1, 1, 0, stream, NULL,
                                 reinterpret_cast<void**>(&config)));

  HIPCHECK(hipStreamSynchronize(stream));

  HIPCHECK(hipStreamDestroy(stream));

  HIPCHECK(hipModuleUnload(Module));

  HIPCHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));

  for (uint32_t i = 0; i < LEN; i++) {
     REQUIRE(A[i] == B[i]);
  }
  HipTest::freeArrays<float>(Ad, Bd, nullptr,
                             A, B, nullptr,
                             false);
}

/*
Thread class inherited from std::thread
*/
struct joinable_thread : std::thread {
  template <class... Xs>
  joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) {} // NOLINT

  joinable_thread& operator=(joinable_thread&& other) = default;
  joinable_thread(joinable_thread&& other)      = default;

  ~joinable_thread() {
    if (this->joinable())
      this->join();
  }
};

/*
This API is triggered form the test case where in
1. Creates the thread object.
2. Loops through the number of GPUs and launches multiple threads.
*/
static void run_multi_threads(uint32_t n, const std::vector<char>& buffer) {
  int numDevices = 0;
  HIPCHECK(hipGetDeviceCount(&numDevices));
  bool testResult = false;
  std::vector<joinable_thread> threads;

  for (int deviceNo=0; deviceNo < numDevices; ++deviceNo) {
    for (uint32_t i = 0; i < n; i++) {
    threads.emplace_back(std::thread{[&, buffer] {
    run(buffer, deviceNo, std::ref(testResult));
    }});
    }
  }
}
/*
The testcase verifies the multithreaded funtionality on MGPU system
1. Loads the kernel file by calling load_file API
2. Gets the host thread count
3. Creates multiple threads in parallel where in each thread initializes
   2 device variables and loads the kernel using hipModuleLoadData API.
   The kernel copies the data from one variable to another.Then the thread
   validates both the variables.
*/
TEST_CASE("Unit_hipModuleLoadData_MGpuMultiThread") {
  auto buffer = load_file();
  auto file_size = buffer.size() / (1024 * 1024);
  auto thread_count =  HipTest::getHostThreadCount(file_size + 10);
  run_multi_threads(thread_count, buffer);
}
