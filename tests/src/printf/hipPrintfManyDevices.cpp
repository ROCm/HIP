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

size_t get_things_size(uint threads_per_device, uint num_devices) {
  DECLARE_DATA();
  const char *msg[] = {msg_short, msg_long1, msg_long2};
  uint num_threads = threads_per_device * num_devices;
  size_t size = 0;

  for(auto str: msg) {
    size += strlen(str) + 1;
  }

  size *= num_threads;
  size += ((threads_per_device + 2) / 3) * num_devices * (strlen(msg_short) + 1);

  return size;
}

int main() {
  uint num_blocks = 14;
  uint threads_per_block = 250;
  uint threads_per_device = num_blocks * threads_per_block;

  CaptureStream capture(stdout);

  int num_devices = 0;
  hipGetDeviceCount(&num_devices);
#ifdef __HIP_PLATFORM_NVIDIA__
  // By default, Cuda has different printf ring buffer size in different GPUs(or ENVs).
  // For example, A100 has 7M, Quadro RTX 5000 has 1.5M, GeForce RTX 2070 Supper has 1.3M in tests.
  // We have to detect, compare and set it
  size_t size = get_things_size(threads_per_device, num_devices);
  size_t size_expected = size * 4;  // Cuda printf buffer format is unknown, but test shows 4 times can work here.
  size_t size_current = 0;
  HIPCHECK(hipDeviceGetLimit(&size_current, hipLimitPrintfFifoSize));
  printf("things size = %zu, expected %zu, current %zu\n", size, size_expected, size_current);

  if(size_current < size_expected) {
    HIPCHECK(hipDeviceSetLimit(hipLimitPrintfFifoSize, size_expected));
  }
#endif
  capture.Begin();
  for (int i = 0; i != num_devices; ++i) {
    hipSetDevice(i);
    hipLaunchKernelGGL(print_things, dim3(num_blocks), dim3(threads_per_block),
                       0, 0);
    hipDeviceSynchronize();
  }
  capture.End();

  std::string data = capture.getData();
  std::stringstream dataStream;
  dataStream << data;

  std::map<std::string, int> linecount;
  for (std::string line; std::getline(dataStream, line);) {
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
