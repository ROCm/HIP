/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */

#include "test_common.h"
#include "printf_common.h"
#include <vector>

// Global string constants don't work inside device functions, so we
// use a macro to repeat the declaration in host and device contexts.
DECLARE_DATA();

__global__ void kernel_uniform0(int *retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  retval[tid] = printf("Hello World\n");
}

static void test_uniform0(int *retval, uint num_blocks,
                          uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_uniform0, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
    HIPASSERT(retval[ii] == strlen("Hello World\n"));
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 1);
  HIPASSERT(linecount["Hello World"] == num_threads);
}

__global__ void kernel_uniform1(int *retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  retval[tid] = printf("Six times Eight is %d\n", 42);
}

static void test_uniform1(int *retval, uint num_blocks,
                          uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_uniform1, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
    HIPASSERT(retval[ii] == strlen("Six times Eight is 42") + 1);
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 1);
  HIPASSERT(linecount["Six times Eight is 42"] == num_threads);
}

__global__ void kernel_divergent0(int *retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  retval[tid] = printf("Thread ID: %d\n", tid);
}

static void test_divergent0(int *retval, uint num_blocks,
                            uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_divergent0, dim3(num_blocks),
                     dim3(threads_per_block), 0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != 10; ++ii) {
    HIPASSERT(retval[ii] == 13);
  }

  for (uint ii = 10; ii != num_threads; ++ii) {
    HIPASSERT(retval[ii] == 14);
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::vector<uint> threadIds;
  for (std::string line; std::getline(dataStream, line);) {
    auto pos = line.find(':');
    HIPASSERT(line.substr(0, pos) == "Thread ID");
    threadIds.push_back(std::stoul(line.substr(pos + 2)));
  }

  std::sort(threadIds.begin(), threadIds.end());
  HIPASSERT(threadIds.size() == num_threads);
  HIPASSERT(threadIds.back() == num_threads - 1);
}

__global__ void kernel_divergent1(int *retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (tid % 2) {
    retval[tid] = printf("Hello World\n");
  } else {
    retval[tid] = -1;
  }
}

static void test_divergent1(int *retval, uint num_blocks,
                            uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_divergent1, dim3(num_blocks),
                     dim3(threads_per_block), 0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
    if (ii % 2) {
      HIPASSERT(retval[ii] == strlen("Hello World\n"));
    } else {
      HIPASSERT(retval[ii] == -1);
    }
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 1);
  HIPASSERT(linecount["Hello World"] == num_threads / 2);
}

__global__ void kernel_series(int *retval) {
  DECLARE_DATA();

  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int result = 0;

  result += printf("%s\n", msg_long1);
  result += printf("%s\n", msg_short);
  result += printf("%s\n", msg_long2);

  retval[tid] = result;
}

static void test_series(int *retval, uint num_blocks, uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_series, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
    HIPASSERT(retval[ii] ==
              strlen(msg_long1) + strlen(msg_short) + strlen(msg_long2) + 3);
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[msg_long1] == num_threads);
  HIPASSERT(linecount[msg_long2] == num_threads);
  HIPASSERT(linecount[msg_short] == num_threads);
}

__global__ void kernel_divergent_loop() {
  DECLARE_DATA();

  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int result = 0;

  for (int i = 0; i <= tid; ++i) {
    printf("%d\n", i);
  }
}

static void test_divergent_loop(uint num_blocks, uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;

  capture.Begin();
  hipLaunchKernelGGL(kernel_divergent_loop, dim3(num_blocks), dim3(threads_per_block),
                     0, 0);
  hipStreamSynchronize(0);
  capture.End();

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<int, int> count;
  while (true) {
    int i;
    dataStream >> i;
    if (dataStream.fail())
      break;
    count[i]++;
  }

  HIPASSERT(count.size() == num_threads);
  for (int i = 0; i != num_threads; ++i) {
    HIPASSERT(count[i] == num_threads - i);
  }
}

int main() {
  uint num_blocks = 1;
  uint threads_per_block = 64;
  uint num_threads = num_blocks * threads_per_block;

  void *retval_void;
  HIPCHECK(hipHostMalloc(&retval_void, 4 * num_threads));
  auto retval = reinterpret_cast<int *>(retval_void);

  test_uniform0(retval, num_blocks, threads_per_block);
  test_uniform1(retval, num_blocks, threads_per_block);
  test_divergent0(retval, num_blocks, threads_per_block);
  test_divergent1(retval, num_blocks, threads_per_block);
  test_series(retval, num_blocks, threads_per_block);
  test_divergent_loop(num_blocks, threads_per_block);

  passed();
}
