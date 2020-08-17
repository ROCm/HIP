/*
 Copyright (c) 2015-2020 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../../src/test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include <iostream>
#include <chrono>
#include "test_common.h"

using namespace std;

#define BufSize 0x1000
#define Iterations 0x100
#define TotalStreams 4
#define TotalBufs 4


class hipPerfStreamCreateCopyDestroy {
  private:
    unsigned int numBuffers_;
    unsigned int numStreams_;
    const size_t totalStreams_[TotalStreams];
    const size_t totalBuffers_[TotalBufs];
  public:
    hipPerfStreamCreateCopyDestroy() : numBuffers_(0), numStreams_(0),
                                       totalStreams_{1, 2, 4, 8},
                                       totalBuffers_{1, 100, 1000, 5000} {};
    ~hipPerfStreamCreateCopyDestroy() {};
    void open(int deviceID);
    void run(unsigned int testNumber);
};

void hipPerfStreamCreateCopyDestroy::open(int deviceId) {
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

void hipPerfStreamCreateCopyDestroy::run(unsigned int testNumber) {
  numStreams_ = totalStreams_[testNumber % TotalStreams];
  size_t iter = Iterations / (numStreams_ * ((size_t)1 << (testNumber / TotalBufs + 1)));
  hipStream_t streams[numStreams_];

  numBuffers_ = totalBuffers_[testNumber / TotalBufs];
  float* dSrc[numBuffers_];
  size_t nBytes = BufSize * sizeof(float);

  for (size_t b = 0; b < numBuffers_; ++b) {
    HIPCHECK(hipMalloc(&dSrc[b], nBytes));
  }

  float* hSrc;
  hSrc = new float[nBytes];
  HIPCHECK(hSrc == 0 ? hipErrorOutOfMemory : hipSuccess);
  for (size_t i = 0; i < BufSize; i++) {
    hSrc[i] = 1.618f + i;
  }

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t s = 0; s < numStreams_; ++s) {
      HIPCHECK(hipStreamCreate(&streams[s]));
    }

    for (size_t s = 0; s < numStreams_; ++s) {
      for (size_t b = 0; b < numBuffers_; ++b) {
        HIPCHECK(hipMemcpyWithStream(dSrc[b], hSrc, nBytes, hipMemcpyHostToDevice, streams[s]));
      }
    }

    for (size_t s = 0; s < numStreams_; ++s) {
      HIPCHECK(hipStreamDestroy(streams[s]));
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  auto time = static_cast<float>(diff.count() * 1000 / (iter * numStreams_));

  cout << "Create+Copy+Destroy time for " << numStreams_ << " streams and "
       << setw(4) << numBuffers_ << " buffers " << " and " << setw(4)
       << iter << " iterations " << time << " (ms) " << endl;

  delete [] hSrc;
  for (size_t b = 0; b < numBuffers_; ++b) {
    HIPCHECK(hipFree(dSrc[b]));
  }
}

int main(int argc, char* argv[]) {
  hipPerfStreamCreateCopyDestroy streamCCD;

  int deviceId = 0;
  streamCCD.open(deviceId);

  for (auto testCase = 0; testCase < TotalStreams * TotalBufs; testCase++) {
    streamCCD.run(testCase);
  }

  passed();
}
