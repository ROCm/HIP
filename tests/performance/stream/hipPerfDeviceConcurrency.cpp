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

typedef struct {
  double x;
  double y;
  double width;
} coordRec;

static coordRec coords[] = {
    {0.0, 0.0, 0.00001},         // All black
};

static unsigned int numCoords = sizeof(coords) / sizeof(coordRec);

__global__ void mandelbrot(uint *out, uint width, float xPos,  float yPos, float xStep,
                            float yStep, uint maxIter) {

  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  int i = tid % width;
  int j = tid / width;
  float x0 = (float)(xPos + xStep*i);
  float y0 = (float)(yPos + yStep*j);

  float x = x0;
  float y = y0;

  uint iter = 0;
  float tmp;
  for (iter = 0; (x*x + y*y <= 4.0f) && (iter < maxIter); iter++) {
    tmp = x;
    x = fma(-y,y,fma(x,x,x0));
    y = fma(2.0f*tmp,y,y0);
  }

  out[tid] = iter;
};

class hipPerfDeviceConcurrency {
  public:
  hipPerfDeviceConcurrency();
  ~hipPerfDeviceConcurrency();

  void setNumGpus(unsigned int num) {
    numDevices = num;
  }
  unsigned int getNumGpus() {
    return numDevices;
  }

  void open(void);
  void close(void);
  void run(unsigned int testCase, int numGpus);

  private:
  void setData(void *ptr, unsigned int value);
  void checkData(uint *ptr);

  unsigned int numDevices;
  unsigned int width_;
  unsigned int bufSize;
  unsigned int coordIdx;
  unsigned long long totalIters = 0;
};


hipPerfDeviceConcurrency::hipPerfDeviceConcurrency() {}

hipPerfDeviceConcurrency::~hipPerfDeviceConcurrency() {}

void hipPerfDeviceConcurrency::open(void) {


  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  setNumGpus(nGpu);
  if (nGpu < 1) {
  std::cout << "info: didn't find any GPU! skipping the test!\n";
  passed();
  }


}


void hipPerfDeviceConcurrency::close() {
}

void hipPerfDeviceConcurrency::run(unsigned int testCase, int numGpus) {


  static int deviceId;
  uint * hPtr[numGpus];
  uint * dPtr[numGpus];
  hipStream_t streams[numGpus];
  int numCUs[numGpus];
  unsigned int maxIter[numGpus];
  unsigned long long expectedIters[numGpus];

  int threads, threads_per_block, blocks;
  float xStep, yStep, xPos, yPos;

  for(int i = 0; i < numGpus; i++) {

  if(testCase != 0) {
    deviceId = i;
  }

  HIPCHECK(hipSetDevice(deviceId));

  hipDeviceProp_t props = {0};
  HIPCHECK(hipGetDeviceProperties(&props, i));

  if (testCase != 0) {
  std::cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name
            << " with " << props.multiProcessorCount << " CUs" << " and device ID: "
            << i << std::endl;
  }

  numCUs[i] = props.multiProcessorCount;
  int clkFrequency = 0;
  HIPCHECK(hipDeviceGetAttribute(&clkFrequency, hipDeviceAttributeClockRate, i));

  clkFrequency =(unsigned int)clkFrequency/1000;

  // Maximum iteration count
  // maxIter = 8388608 * (engine_clock / 1000).serial execution
  maxIter[i] = (unsigned int)(((8388608 * ((float)clkFrequency / 1000)) * numCUs[i]) / 128);
  maxIter[i] = (maxIter[i] + 15) & ~15;

  // Width is divisible by 4 because the mandelbrot kernel processes 4 pixels at once.
  width_ = 256;

  bufSize = width_ * width_ * sizeof(uint);

  // Create streams for concurrency
  HIPCHECK(hipStreamCreate(&streams[i]));

  // Allocate memory on the host and device
  HIPCHECK(hipHostMalloc((void **)&hPtr[i], bufSize, hipHostMallocDefault));
  setData(hPtr[i], 0xdeadbeef);
  HIPCHECK(hipMalloc((uint **)&dPtr[i], bufSize))

  // Prepare kernel launch parameters
  threads = (bufSize/sizeof(uint));
  threads_per_block  = 64;
  blocks = (threads/threads_per_block) + (threads % threads_per_block);

  coordIdx = testCase % numCoords;
  xStep = (float)(coords[coordIdx].width / (double)width_);
  yStep = (float)(-coords[coordIdx].width / (double)width_);
  xPos = (float)(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
  yPos = (float)(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

  // Copy memory from host to device
  HIPCHECK(hipMemcpy(dPtr[i], hPtr[i], bufSize, hipMemcpyHostToDevice));

  }

  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();

  for(int i = 0; i < numGpus; i++) {

  if(testCase != 0) {
    deviceId = i;
  }

  HIPCHECK(hipSetDevice(deviceId));

  hipLaunchKernelGGL(mandelbrot, dim3(blocks), dim3(threads_per_block), 0, streams[i],
                      dPtr[i], width_, xPos, yPos, xStep, yStep, maxIter[i]);

  }

  for(int i = 0; i < numGpus; i++) {
    HIPCHECK(hipStreamSynchronize(0));
  }


  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  for(int i = 0; i < numGpus; i++) {

  if(testCase != 0) {
    deviceId = i;
  }
  HIPCHECK(hipSetDevice(deviceId));

  // Copy data back from device to the host
  HIPCHECK(hipMemcpy(hPtr[i], dPtr[i], bufSize, hipMemcpyDeviceToHost));

  checkData(hPtr[i]);
  expectedIters[i] = width_ * width_ * (unsigned long long) maxIter[i];

  if (testCase != 0) {
    checkData(hPtr[i]);
    if(totalIters != expectedIters[i]) {
      std::cout << "Incorrect iteration count detected" << std::endl;
    }
  }


  HIPCHECK(hipStreamDestroy(streams[i]));

  // Free host and device memory
  HIPCHECK(hipFree(hPtr[i]));
  HIPCHECK(hipFree(dPtr[i]));
  }

  if (testCase != 0) {
  std::cout << '\n' << "Measured time for kernel computation on " << numGpus << " device (s): "
            << all_kernel_time.count() << " (s) " << '\n' << std::endl;
  }

  if(testCase == 0) {
    deviceId++;
  }


}


void hipPerfDeviceConcurrency::setData(void *ptr, unsigned int value) {
  unsigned int *ptr2 = (unsigned int *)ptr;
  for (unsigned int i = 0; i < width_ * width_ ; i++) {
      ptr2[i] = value;
  }
}


void hipPerfDeviceConcurrency::checkData(uint *ptr) {
  totalIters = 0;
  for (unsigned int i = 0; i < width_ * width_; i++) {
    totalIters += ptr[i];
  }
}


int main(int argc, char* argv[]) {
  hipPerfDeviceConcurrency deviceConcurrency;

  deviceConcurrency.open();

  int nGpu = deviceConcurrency.getNumGpus();

  // testCase = 0 refers to warmup kernel run
  int testCase = 0;

  for (int i = 0; i < nGpu; i++) {
    // Warm-up kernel on all devices
    deviceConcurrency.run(testCase, 1);
  }

  // Time for kernel on 1 device
  deviceConcurrency.run(++testCase, 1);

  // Time for kernel on all available devices
  deviceConcurrency.run(++testCase, nGpu);

  passed();
}
