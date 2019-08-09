/*
Copyright (c) 2015-2019 Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

/*
 * Test for checking the functionality of
 * hipStreamCreateWithPriority
 */

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#include <getopt.h>
#endif

#include "hip/hip_runtime.h"
#include "test_common.h"

__global__ void dummy(int* output, int init_value, int multiplier, int num_iters) {
  *output = init_value;
  for (int i = 0; i < num_iters; ++i) {
    *output+=*output*multiplier;
  }
}

void run(const int thread_id, const bool descending, const int num_streams_per_priority) {

  int lp, gp;
  HIPCHECK(hipDeviceGetStreamPriorityRange(&lp, &gp));
#ifdef DEBUG_OUTPUT
  printf("Stream priority range: %d to %d\n", lp, gp);
#endif
  int start_priority, end_priority, bump;
  if (descending) {
    start_priority = gp;
    end_priority = lp;
    bump = 1;
  }
  else {
    start_priority = lp;
    end_priority = gp;
    bump = -1; 
  }

  struct gpu_resources {
    hipStream_t s = 0;
    int stream_priority;
    int* output = nullptr;
    gpu_resources(int stream_priority) : stream_priority(stream_priority) {
      HIPCHECK(hipStreamCreateWithPriority(&s , hipStreamDefault, stream_priority));
      HIPCHECK(hipMalloc(&output, sizeof(output))); 
    }
    gpu_resources(gpu_resources&& t) : 
       s(t.s), stream_priority(t.stream_priority),
       output(t.output) {
      t.s = 0;
      t.output = nullptr;
    }
    ~gpu_resources() {
      if (s)      HIPCHECK(hipStreamDestroy(s));
      if (output) HIPCHECK(hipFree(output));
    }
  };

  std::vector<gpu_resources> v_gpu_resources;
  int num_streams = 0;
  for (int p = start_priority; ; p+=bump) {
      for (int np = 0; np < num_streams_per_priority; ++np, ++num_streams) {
#ifdef DEBUG_OUTPUT
          printf("Thread %d: creating stream %d with priority %d\n", thread_id, num_streams, p);
#endif
          gpu_resources r(p);
          hipLaunchKernelGGL(dummy, dim3(1), dim3(1), 0, r.s, r.output, 0, 1, 1000);
          v_gpu_resources.push_back(std::move(r));
      }
      if (p == end_priority)
        break;
  }
  for (auto& r : v_gpu_resources) {
    HIPCHECK(hipStreamSynchronize(r.s)); 
  }
}

int main(int argc, char* argv[]) {

  int num_threads = 8;
  int num_streams_per_priority = 24;

#ifdef _GNU_SOURCE
  while (1) {
    static struct option opts[] = {
      {"streams", required_argument, 0, 's'},
      {"threads", required_argument, 0, 't'},
      {0, 0, 0, 0}
    };
    
    int parse_idx = 0;
    int s = getopt_long(argc, argv, "s:t:", opts, &parse_idx);
    if (s == -1) break;

    switch (s) {
      case 's':
        num_streams_per_priority = std::stoi(std::string(optarg));
        break;
      case 't':
        num_threads = std::stoi(std::string(optarg));
        break;
      default:
        abort();
    }
  }
#endif // _GNU_SOURCE

  std::vector<std::thread> threads;
  bool descending = true;
  for (auto i = 0; i < num_threads; ++i) {
    threads.push_back(std::thread(run, i, descending, num_streams_per_priority));
    descending = !descending;
  }
  for (auto& t : threads) {
    t.join();
  }
  passed();
}