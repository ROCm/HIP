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
#include <hip/hip_vector_types.h>
#include <vector>

using namespace std;

#define NUM_TYPES 3
vector<string> types= {"float", "float2", "float4"};
vector<unsigned int> typeSizes = {4, 8, 16};

#define NUM_SIZES 12
vector<unsigned int> sizes = {1,  2,   4,   8,   16,   32,
                              64, 128, 256, 512, 1024, 2048};

#define NUM_BUFS 6
#define MAX_BUFS (1 << (NUM_BUFS - 1))

template <typename T>
__global__ void sampleRate(T * outBuffer, unsigned int inBufSize, unsigned int writeIt,
                           T **inBuffer, int numBufs) {

  uint gid = (blockIdx.x * blockDim.x + threadIdx.x);
  uint inputIdx = gid % inBufSize;

  T tmp = (T)0.0f;
  for(int i = 0; i < numBufs; i++) {
    tmp += *(*(inBuffer+i)+inputIdx);
  }

  if (writeIt*(unsigned int)tmp.x) {
    outBuffer[gid] = tmp;
   }
};

template <typename T>
__global__ void sampleRateFloat(T * outBuffer, unsigned int inBufSize, unsigned int writeIt,
                                T ** inBuffer, int numBufs) {

  uint gid = (blockIdx.x * blockDim.x + threadIdx.x);
  uint inputIdx = gid % inBufSize;

  T tmp = (T)0.0f;

  for(int i = 0; i < numBufs; i++) {
    tmp += *((*inBuffer+i)+inputIdx);
  }

  if (writeIt*(unsigned int)tmp) {
    outBuffer[gid] = tmp;
  }
};

class hipPerfSampleRate {
  public:
  hipPerfSampleRate();
  ~hipPerfSampleRate();

  void open(void);
  void run(unsigned int testCase);
  void close(void);

  // array of funtion pointers
  typedef void (hipPerfSampleRate::*funPtr)(void * outBuffer, unsigned int
    inBufSize, unsigned int writeIt, void **inBuffer, int numBufs, int grids, int blocks,
    int threads_per_block);

  // Wrappers
  void float_kernel(void * outBuffer, unsigned int
    inBufSize, unsigned int writeIt, void **inBuffer, int numBufs, int grids, int blocks,
    int threads_per_block);

  void float2_kernel(void * outBuffer, unsigned int
    inBufSize, unsigned int writeIt, void **inBuffer, int numBufs, int grids, int blocks,
    int threads_per_block);

  void float4_kernel(void * outBuffer, unsigned int
    inBufSize, unsigned int writeIt, void **inBuffer, int numBufs, int grids, int blocks,
    int threads_per_block);

  private:
  void setData(void *ptr, unsigned int value);
  void checkData(uint *ptr);

  unsigned int width_;
  unsigned int bufSize_;
  unsigned long long totalIters = 0;
  int numCUs;

  unsigned int outBufSize_;
  static const unsigned int MAX_ITERATIONS = 25;
  unsigned int numBufs_;
  unsigned int typeIdx_;
};


hipPerfSampleRate::hipPerfSampleRate() {}

hipPerfSampleRate::~hipPerfSampleRate() {}

void hipPerfSampleRate::open(void) {

  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
   if (nGpu < 1) {
  std::cout << "info: didn't find any GPU! skipping the test!\n";
  passed();
  return;
  }

  int deviceId = 0;
  hipDeviceProp_t props = {0};
  props = {0};
  HIPCHECK(hipSetDevice(deviceId));
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));
  std::cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name
    << " with " << props.multiProcessorCount << " CUs" << " and device id: " << deviceId
    << std::endl;
  numCUs = props.multiProcessorCount;
  }


void hipPerfSampleRate::close() {

}


// Wrappers for the kernel launches
void hipPerfSampleRate::float_kernel(void * outBuffer, unsigned int inBufSize,
                                       unsigned int writeIt, void **inBuffer,
                                       int numBufs, int grids, int blocks, int threads_per_block) {

  hipLaunchKernelGGL(sampleRateFloat<float>, dim3(grids, grids, grids), dim3 (blocks), 0, 0,
                      (float*)outBuffer, inBufSize, writeIt, (float**)inBuffer, numBufs);

}

void hipPerfSampleRate::float2_kernel(void * outBuffer, unsigned int inBufSize,
                                       unsigned int writeIt, void **inBuffer,
                                       int grids, int blocks, int threads_per_block, int numBufs) {

  hipLaunchKernelGGL(sampleRate<float2>, dim3(grids, grids, grids), dim3(blocks), 0, 0,
                      (float2 *)outBuffer, inBufSize, writeIt, (float2**)inBuffer, numBufs);
}

void hipPerfSampleRate::float4_kernel(void * outBuffer, unsigned int inBufSize,
                                       unsigned int writeIt, void **inBuffer,
                                       int grids, int blocks, int threads_per_block, int numBufs) {

  hipLaunchKernelGGL(sampleRate<float4>, dim3(grids, grids, grids), dim3(blocks), 0, 0,
                      (float4 *) outBuffer, inBufSize, writeIt, (float4**)inBuffer, numBufs);
}

void hipPerfSampleRate::run(unsigned int test) {

  funPtr p[] = {&hipPerfSampleRate::float_kernel, &hipPerfSampleRate::float2_kernel,
               &hipPerfSampleRate::float4_kernel};

  // We compute a square domain
  width_ = sizes[test % NUM_SIZES];
  typeIdx_ = (test / NUM_SIZES) % NUM_TYPES;
  bufSize_ = width_ * width_ * typeSizes[typeIdx_];
  numBufs_ = (1 << (test / (NUM_SIZES * NUM_TYPES)));

  void *  hOutPtr;
  void *  dOutPtr;
  void *  hInPtr[numBufs_];
  void ** dPtr;
  void *  dInPtr[numBufs_];

 outBufSize_ =
      sizes[NUM_SIZES - 1] * sizes[NUM_SIZES - 1] * typeSizes[NUM_TYPES - 1];

  // Allocate memory on the host and device
  HIPCHECK(hipHostMalloc((void **)&hOutPtr, outBufSize_, hipHostMallocDefault));
  setData((void *)hOutPtr, 0xdeadbeef);
  HIPCHECK(hipMalloc((uint **)&dOutPtr, outBufSize_));

  // Allocate 2D array in Device
   hipMalloc((void **)&dPtr, numBufs_* sizeof(void *));

  for (uint i = 0; i < numBufs_; i++) {
    HIPCHECK(hipHostMalloc((void **)&hInPtr[i], bufSize_, hipHostMallocDefault));
    HIPCHECK(hipMalloc((uint **)&dInPtr[i], bufSize_));
    setData(hInPtr[i], 0x3f800000);
  }

  // Populate array of pointers with array addresses
  hipMemcpy(dPtr, dInPtr, numBufs_* sizeof(void *), hipMemcpyHostToDevice);

  // Copy memory from host to device
  for (uint i = 0; i < numBufs_; i++) {
  HIPCHECK(hipMemcpy(dInPtr[i], hInPtr[i], bufSize_, hipMemcpyHostToDevice));
  }

  HIPCHECK(hipMemcpy(dOutPtr, hOutPtr, outBufSize_, hipMemcpyHostToDevice));

  // Prepare kernel launch parameters
  // outBufSize_/sizeof(uint) - Grid size in 3D
  int grids = 64;
  int blocks = 64;
  int threads_per_block  = 1;

  unsigned int maxIter = MAX_ITERATIONS * (MAX_BUFS / numBufs_);
  unsigned int sizeDW = width_ * width_;
  unsigned int writeIt = 0;

  int idx = 0;

  if (!types[typeIdx_].compare("float")) {
    idx = 0;
  }
  else if(!types[typeIdx_].compare("float2")) {
          idx = 1;
  }
  else if(!types[typeIdx_].compare("float4")) {
          idx = 2;
  }


  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();
  for (uint i = 0; i < maxIter; i++) {
        (this->*p[idx]) ((void *)dOutPtr, sizeDW, writeIt, dPtr, numBufs_, grids, blocks,
                          threads_per_block);
  }

  hipDeviceSynchronize();
  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  double perf = ((double)outBufSize_ * numBufs_ * (double)maxIter * (double)(1e-09)) /
                          all_kernel_time.count();

  cout << "Domain " << sizes[NUM_SIZES - 1] << "x"<< sizes[NUM_SIZES - 1] << " bufs "
       << numBufs_ << " " << types[typeIdx_] << " " << width_<<"x"<<width_<< " (GB/s) "
       << perf << endl;

   HIPCHECK(hipFree(dOutPtr));

   // Free host and device memory
   for (uint i = 0; i < numBufs_; i++) {
    HIPCHECK(hipFree(hInPtr[i]));
    HIPCHECK(hipFree(dInPtr[i]));
    }

   HIPCHECK(hipFree(hOutPtr));
   HIPCHECK(hipFree(dPtr));
}


void hipPerfSampleRate::setData(void *ptr, unsigned int value) {
  unsigned int *ptr2 = (unsigned int *)ptr;
  for (unsigned int i = 0; i < bufSize_ / sizeof(unsigned int); i++) {
      ptr2[i] = value;
  }
}


void hipPerfSampleRate::checkData(uint *ptr) {
  for (unsigned int i = 0; i < outBufSize_ / sizeof(float); i++) {
    if (ptr[i] != (float)numBufs_) {
      cout << "Data validation failed at "<< i << " Got "<< ptr[i] << ", expected "
           << (float)numBufs_;
      break;
          }
  }
}


int main(int argc, char* argv[]) {
  hipPerfSampleRate sampleTypes;

  sampleTypes.open();

  for (unsigned int testCase = 0; testCase < 216 ; testCase+=36) {
    sampleTypes.run(testCase);
  }


  passed();
}
