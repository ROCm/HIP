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

/*
This testcase verifies hipModuleLoad API in multithreaded scenario 
*/
#include <stdio.h>
#include "hip/hip_runtime.h"
#if HT_AMD
#include "hip/hip_ext.h"
#endif
#include <fstream>
#include <algorithm>
#include <atomic>
#include <functional>
#include <vector>
#include <future>
#define THREADS 8
#define MAX_NUM_THREADS 128

#include "hip_test_common.hh"
#include "hip_test_checkers.hh"

#define NUM_GROUPS 1
#define GROUP_SIZE 1
#define WARMUP_RUN_COUNT 10
#define TIMING_RUN_COUNT 100
#define TOTAL_RUN_COUNT WARMUP_RUN_COUNT + TIMING_RUN_COUNT
#define FILENAME "module_kernels.code"
#define kernel_name "EmptyKernel"

/*
This thread function loads the kernel file , synchronizes the threads
and Launches the kernel .
*/
void hipModuleLaunchKernel_enqueue(std::atomic_int* shared, int max_threads) {
    // resources necessary for this thread
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    hipModule_t module;
    hipFunction_t function;

    HIPCHECK(hipModuleLoad(&module, FILENAME));
    HIPCHECK(hipModuleGetFunction(&function, module, kernel_name));

    void* kernel_params = nullptr;

    // synchronize all threads, before running
    shared->fetch_add(1, std::memory_order_release);
    while (max_threads != shared->load(std::memory_order_acquire)) {}

    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        HIPCHECK(hipModuleLaunchKernel(function, 1, 1,
                                       1, 1, 1, 1, 0, stream,
                                       &kernel_params, nullptr));
    }
    HIPCHECK(hipModuleUnload(module));
    HIPCHECK(hipStreamDestroy(stream));
}

/*
thread pool class contains launching the threads using std::async API
with future variable "threads".
The start API Launches the threads and finish API waits for the
thread execution to end.
*/
struct thread_pool {
  explicit thread_pool(int total_threads) : max_threads(total_threads) {
  }
  void start(std::function<void(std::atomic_int*, int)> f) {
    for (int i = 0; i < max_threads; ++i) {
      threads.push_back(std::async(std::launch::async, f,
                                   &shared, max_threads));
    }
  }
  void finish() {
    for (auto&&thread : threads) {
      thread.get();
    }
    threads.clear();
    shared = 0;
  }
  ~thread_pool() {
    finish();
  }
 private:
  std::atomic_int shared {0};
  std::vector<char> buffer;
  std::vector<std::future<void>> threads;
  int max_threads = 1;
};

/*
This testcase verifies the Multithreaded scenario of hipModule API
where in threadpool object is created and the object invokes start API
which launches multiple threads where each thread loads the kernel object
using hipModuleLoad API and launches the kernel in parallel.
*/
TEST_CASE("Unit_hipModuleLoad_MultiThread") {
    int max_threads = min(THREADS * std::thread::hardware_concurrency(),
                          MAX_NUM_THREADS);
    thread_pool task(max_threads);
    task.start(hipModuleLaunchKernel_enqueue);
    task.finish();
}
