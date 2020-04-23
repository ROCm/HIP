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
 * BUILD: %t %s EXCLUDE_HIP_PLATFORM nvcc EXCLUDE_HIP_RUNTIME HCC EXCLUDE_HIP_COMPILER hcc
 * TEST: %t EXCLUDE_HIP_PLATFORM nvcc EXCLUDE_HIP_RUNTIME HCC EXCLUDE_HIP_COMPILER hcc
 * HIT_END
 */

#include "test_common.h"
#include "printf_common.h"
#include <vector>

// Global string constants don't work inside device functions, so we
// use a macro to repeat the declaration in host and device contexts.
DECLARE_DATA();

__global__ void kernel_mixed0(int *retval) {
  DECLARE_DATA();

  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  ulong result = 0;

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
  CaptureStream captured(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  hipLaunchKernelGGL(kernel_mixed0, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  auto CapturedData = captured.getCapturedData();

  for (uint ii = 0; ii != num_threads; ++ii) {
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
  }

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[msg_short] == (num_threads + 2) / 3);
  HIPASSERT(linecount[msg_long1] == (num_threads + 1) / 3);
  HIPASSERT(linecount[msg_long2] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed1(int *retval) {
  DECLARE_DATA();

  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

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
  CaptureStream captured(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  hipLaunchKernelGGL(kernel_mixed1, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  auto CapturedData = captured.getCapturedData();

  for (uint ii = 0; ii != num_threads; ++ii) {
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
  }

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[msg_short] == (num_threads + 2) / 3);
  HIPASSERT(linecount[msg_long1] == (num_threads + 1) / 3);
  HIPASSERT(linecount[msg_long2] == (num_threads + 0) / 3);
}

__global__ void kernel_mixed2(int *retval) {
  DECLARE_DATA();

  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

  // Three different strings. All workitems print all three, but
  // in different orders.
  const char *msg[] = {msg_short, msg_long1, msg_long2};
  retval[tid] =
      printf("%s%s%s\n", msg[tid % 3], msg[(tid + 1) % 3], msg[(tid + 2) % 3]);
}

static void test_mixed2(int *retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  hipLaunchKernelGGL(kernel_mixed2, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  auto CapturedData = captured.getCapturedData();

  for (uint ii = 0; ii != num_threads; ++ii) {
    HIPASSERT(retval[ii] ==
              strlen(msg_short) + strlen(msg_long1) + strlen(msg_long2) + 1);
  }

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
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

  const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  int result = 0;

  result += printf("%s\n", msg_long1);
  if (tid % 3 == 0) {
    result += printf("%s\n", msg_short);
  }
  result += printf("%s\n", msg_long2);

  retval[tid] = result;
}

static void test_mixed3(int *retval, uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);

  uint num_threads = num_blocks * threads_per_block;
  for (uint i = 0; i != num_threads; ++i) {
    retval[i] = 0x23232323;
  }

  hipLaunchKernelGGL(kernel_mixed3, dim3(num_blocks), dim3(threads_per_block),
                     0, 0, retval);
  hipStreamSynchronize(0);
  auto CapturedData = captured.getCapturedData();

  for (uint ii = 0; ii != num_threads; ++ii) {
    if (ii % 3 == 0) {
      HIPASSERT(retval[ii] ==
                strlen(msg_long1) + strlen(msg_short) + strlen(msg_long2) + 3);
    } else {
      HIPASSERT(retval[ii] == strlen(msg_long1) + strlen(msg_long2) + 2);
    }
  }

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }

  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[msg_long1] == num_threads);
  HIPASSERT(linecount[msg_long2] == num_threads);
  HIPASSERT(linecount[msg_short] == (num_threads + 2) / 3);
}

__global__ void kernel_numbers() {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  for (uint i = 0; i != 7; ++i) {
    uint base = tid * 21 + i * 3;
    printf("%d %d %d\n", base, base + 1, base + 2);
  }
}

static void test_numbers(uint num_blocks, uint threads_per_block) {
  CaptureStream captured(stdout);
  uint num_threads = num_blocks * threads_per_block;

  hipLaunchKernelGGL(kernel_numbers, dim3(num_blocks), dim3(threads_per_block),
                     0, 0);
  hipStreamSynchronize(0);
  auto CapturedData = captured.getCapturedData();

  std::vector<uint> points;
  while (true) {
    uint i;
    CapturedData >> i;
    if (CapturedData.fail())
      break;
    points.push_back(i);
  }

  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  HIPASSERT(points.size() == 21 * num_threads);
  HIPASSERT(points.back() == 21 * num_threads - 1);

  passed();
}

int main(int argc, char **argv) {
  uint num_blocks = 150;
  uint threads_per_block = 250;
  uint num_threads = num_blocks * threads_per_block;

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
