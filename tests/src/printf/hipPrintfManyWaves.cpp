/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include "printf_common.h"
#include <vector>
#include <algorithm>

// Global string constants don't work inside device functions, so we
// use a macro to repeat the declaration in host and device contexts.
DECLARE_DATA();

__global__ void kernel_mixed0(int *retval) {
  DECLARE_DATA();

  uint tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Three strings passed as divergent values to the same hostcall.
  const char *msg;
  switch (tid % 3) {
  case 0:
    msg = msg_short;
    break;
  case 1:
    msg = msg_long1;
    break;
  case 2:
    msg = msg_long2;
    break;
  }

  retval[tid] = printf("%s\n", msg);
}

static void test_mixed0(int *retval, uint num_blocks, uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_mixed0, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    switch (ii % 3) {
    case 0:
      HIPASSERT(retval[ii] == strlen(msg_short) + 1);
      break;
    case 1:
      HIPASSERT(retval[ii] == strlen(msg_long1) + 1);
      break;
    case 2:
      HIPASSERT(retval[ii] == strlen(msg_long2) + 1);
      break;
    }
#else
    HIPASSERT(retval[ii] == 1);
#endif
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[msg_short] == (num_threads + 2) / 3);
  HIPASSERT(linecount[msg_long1] == (num_threads + 1) / 3);
  HIPASSERT(linecount[msg_long2] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed1(int *retval) {
  DECLARE_DATA();

  const uint tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Three strings passed to divergent hostcalls.
  switch (tid % 3) {
  case 0:
    retval[tid] = printf("%s\n", msg_short);
    break;
  case 1:
    retval[tid] = printf("%s\n", msg_long1);
    break;
  case 2:
    retval[tid] = printf("%s\n", msg_long2);
    break;
  }
}

static void test_mixed1(int *retval, uint num_blocks, uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_mixed1, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    switch (ii % 3) {
    case 0:
      HIPASSERT(retval[ii] == strlen(msg_short) + 1);
      break;
    case 1:
      HIPASSERT(retval[ii] == strlen(msg_long1) + 1);
      break;
    case 2:
      HIPASSERT(retval[ii] == strlen(msg_long2) + 1);
      break;
    }
#else
    HIPASSERT(retval[ii] == 1);
#endif
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[msg_short] == (num_threads + 2) / 3);
  HIPASSERT(linecount[msg_long1] == (num_threads + 1) / 3);
  HIPASSERT(linecount[msg_long2] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed2(int *retval) {
  DECLARE_DATA();

  const uint tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Three different strings. All workitems print all three, but
  // in different orders.
  const char *msg[] = {msg_short, msg_long1, msg_long2};
  retval[tid] =
      printf("%s%s%s\n", msg[tid % 3], msg[(tid + 1) % 3], msg[(tid + 2) % 3]);
}

static void test_mixed2(int *retval, uint num_blocks, uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_mixed2, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    HIPASSERT(retval[ii] ==
              strlen(msg_short) + strlen(msg_long1) + strlen(msg_long2) + 1);
#else
    HIPASSERT(retval[ii] == 3);
#endif
  }

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
    linecount[line]++;
  }

  std::string str1 =
      std::string(msg_short) + std::string(msg_long1) + std::string(msg_long2);
  std::string str2 =
      std::string(msg_long1) + std::string(msg_long2) + std::string(msg_short);
  std::string str3 =
      std::string(msg_long2) + std::string(msg_short) + std::string(msg_long1);

  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[str1] == (num_threads + 2) / 3);
  HIPASSERT(linecount[str2] == (num_threads + 1) / 3);
  HIPASSERT(linecount[str3] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed3(int *retval) {
  DECLARE_DATA();

  const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  int result = 0;

  result += printf("%s\n", msg_long1);
  if (tid % 3 == 0) {
    result += printf("%s\n", msg_short);
  }
  result += printf("%s\n", msg_long2);

  retval[tid] = result;
}

size_t get_mixed3_size(uint num_threads) {
  DECLARE_DATA();
  const char *msg[] = {msg_long1, msg_long2};
  size_t size = 0;

  for(auto str: msg) {
    size += strlen(str) + 1;
  }

  size *= num_threads;
  size += ((num_threads + 2) / 3) * (strlen(msg_short) + 1);

  return size;
}

static void test_mixed3(int *retval, uint num_blocks, uint threads_per_block) {
  CaptureStream capture(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  capture.Begin();
  hipLaunchKernelGGL(kernel_mixed3, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  capture.End();

  for (uint ii = 0; ii != num_threads; ++ii) {
#ifdef __HIP_PLATFORM_AMD__
    if (ii % 3 == 0) {
      HIPASSERT(retval[ii] ==
                strlen(msg_long1) + strlen(msg_short) + strlen(msg_long2) + 3);
    } else {
      HIPASSERT(retval[ii] == strlen(msg_long1) + strlen(msg_long2) + 2);
    }
#else
    HIPASSERT(retval[ii] == (ii % 3 ? 2 : 3));
#endif
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
  HIPASSERT(linecount[msg_short] == (num_threads + 2) / 3);
}

__global__ void kernel_numbers() {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (uint i = 0; i != 7; ++i) {
    uint base = tid * 21 + i * 3;
    printf("%d %d %d\n", base, base + 1, base + 2);
  }
}

size_t get_numbers_size(uint num_threads) {
  char buf[100] = { 0 };
  size_t size = 0;
  for (uint tid = 0; tid < num_threads; tid++) {
    for (uint i = 0; i != 7; ++i) {
      uint base = tid * 21 + i * 3;
      size += snprintf(buf, 100, "%d %d %d\n", base, base + 1, base + 2);
    }
  }
  return size;
}

static void test_numbers(uint num_blocks, uint threads_per_block) {
  CaptureStream capture(stdout);
  uint num_threads = num_blocks * threads_per_block;

  capture.Begin();
  hipLaunchKernelGGL(kernel_numbers, dim3(num_blocks), dim3(threads_per_block),
                     0, 0);
  hipStreamSynchronize(0);
  capture.End();

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::vector<uint> points;
  while (true) {
    uint i;
    dataStream >> i;
    if (dataStream.fail())
      break;
    points.push_back(i);
  }

  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  HIPASSERT(points.size() == 21 * num_threads);
  HIPASSERT(points.back() == 21 * num_threads - 1);
}

int main(int argc, char **argv) {
  uint num_blocks = 150;
  uint threads_per_block = 250;
  uint num_threads = num_blocks * threads_per_block;
#ifdef __HIP_PLATFORM_NVIDIA__
  // By default, Cuda has different printf ring buffer size in different GPUs(or ENVs).
  // For example, A100 has 7M, Quadro RTX 5000 has 1.5M, GeForce RTX 2070 Supper has 1.3M in tests.
  // We have to detect, compare and set it
  size_t size_mixed3 = get_mixed3_size(num_threads);
  size_t size_numbers = get_numbers_size(num_threads);
  size_t size_max = size_mixed3 >= size_numbers ? size_mixed3 : size_numbers;  // Max size
  size_t size_expected = size_max * 10;  // Cuda printf buffer format is unknown, but test shows 10 times can work here.
  size_t size_current = 0;
  HIPCHECK(hipDeviceGetLimit(&size_current, hipLimitPrintfFifoSize));
  printf("size_mixed3 = %zu, size_numbers = %zu\n", size_mixed3, size_numbers);
  printf("max size = %zu, expected %zu, current %zu\n", size_max, size_expected, size_current);

  if(size_current < size_expected) {
    HIPCHECK(hipDeviceSetLimit(hipLimitPrintfFifoSize, size_expected));
  }
#endif
  void *retval_void;
  HIPCHECK(hipHostMalloc(&retval_void, 4 * num_threads));
  auto retval = reinterpret_cast<int *>(retval_void);

  test_mixed0(retval, num_blocks, threads_per_block);
  test_mixed1(retval, num_blocks, threads_per_block);
  test_mixed2(retval, num_blocks, threads_per_block);
  test_mixed3(retval, num_blocks, threads_per_block);
  test_numbers(num_blocks, threads_per_block);
  passed();
}
