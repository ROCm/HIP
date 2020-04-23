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

DECLARE_DATA();

__global__ void print_things() {
  DECLARE_DATA();

  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  const char *msg[] = {msg_short, msg_long1, msg_long2};

  printf("%s\n", msg[tid % 3]);
  if (tid % 3 == 0)
    printf("%s\n", msg_short);
  printf("%s\n", msg[(tid + 1) % 3]);
  printf("%s\n", msg[(tid + 2) % 3]);
}

int main() {
  uint num_blocks = 14;
  uint threads_per_block = 250;
  uint threads_per_device = num_blocks * threads_per_block;

  int num_devices = 0;
  hipGetDeviceCount(&num_devices);

  CaptureStream captured(stdout);
  for (int i = 0; i != num_devices; ++i) {
    hipSetDevice(i);
    hipLaunchKernelGGL(print_things, dim3(num_blocks), dim3(threads_per_block),
                       0, 0);
    hipDeviceSynchronize();
  }
  auto CapturedData = captured.getCapturedData();

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(CapturedData, line);) {
    linecount[line]++;
  }

  uint num_threads = threads_per_device * num_devices;
  HIPASSERT(linecount.size() == 3);
  HIPASSERT(linecount[msg_long1] == num_threads);
  HIPASSERT(linecount[msg_long2] == num_threads);
  HIPASSERT(linecount[msg_short] ==
            num_threads + ((threads_per_device + 2) / 3) * num_devices);

  passed();
}
