/*
 Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../../src/test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <iostream>
#include <chrono>

#define NUM_SIZE 8
#define NUM_ITER 0x40000


using namespace std;

class hipPerfMemcpy {
  private:
    unsigned int numBuffers_;
    size_t totalSizes_[NUM_SIZE];
    void setHostBuffer(int *A, int val, size_t size);
  public:
    hipPerfMemcpy();
    ~hipPerfMemcpy() {};
    void open(int deviceID);
    void run(unsigned int testNumber);
};

hipPerfMemcpy::hipPerfMemcpy() : numBuffers_(0) {
  for (int i = 0; i < NUM_SIZE; i++) {
    totalSizes_[i] = 1 << (i + 6);
  }
};

void hipPerfMemcpy::setHostBuffer(int *A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (int i = 0; i < len; i++) {
    A[i] = val;
  }
}

void hipPerfMemcpy::open(int deviceId) {
  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    std::cout << "info: didn't find any GPU! skipping the test!\n";
    passed();
    return;
  }

  HIPCHECK(hipSetDevice(deviceId));
  hipDeviceProp_t props = {0};
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));
  std::cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name
    << " with " << props.multiProcessorCount << " CUs" << " and device id: " << deviceId  << std::endl;
}

void hipPerfMemcpy::run(unsigned int testNumber) {
  int *A, *Ad;
  A = new int[totalSizes_[testNumber]];
  setHostBuffer(A, 1, totalSizes_[testNumber]);
  hipMalloc(&Ad, totalSizes_[testNumber]);

  auto start = chrono::steady_clock::now();

  for (int j = 0; j < NUM_ITER; j++) {
    hipMemcpy(Ad, A, totalSizes_[testNumber], hipMemcpyHostToDevice);
  }

  hipDeviceSynchronize();

  auto end = chrono::steady_clock::now();
  chrono::duration<double, micro> diff = end - start;

  cout << "hipPerfMemcpy[" << testNumber << "] " << "Host to Device copy took "
      << diff.count() / NUM_ITER << " us for memory size of " << totalSizes_[testNumber]
      << " Bytes" << endl;

  delete [] A;
  HIPCHECK(hipFree(Ad));

}


int main() {
  hipPerfMemcpy hipPerfMemcpy;

  int deviceId = 0;
  hipPerfMemcpy.open(deviceId);

  for (auto testCase = 0; testCase < NUM_SIZE; testCase++) {
    hipPerfMemcpy.run(testCase);
  }

  passed();

}
