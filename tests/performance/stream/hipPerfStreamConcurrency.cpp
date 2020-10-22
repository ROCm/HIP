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

#include <iostream>
#include <chrono>
#include "test_common.h"
#include <hip/hip_vector_types.h>


typedef struct {
  double x;
  double y;
  double width;
} coordRec;

static coordRec coords[] = {
    {0.0, 0.0, 0.00001},  // All black
};


static unsigned int numCoords = sizeof(coords) / sizeof(coordRec);

__global__ void mandelbrot(uint *out, uint width, float xPos, float yPos,
         float xStep, float yStep, uint maxIter) {


  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  int i = tid % (width/4);
  int j = tid / (width/4);
  int4 veci = make_int4(4*i, 4*i+1, 4*i+2, 4*i+3);
  int4 vecj = make_int4(j, j, j, j);
  float4 x0;
  x0.data[0] = (float)(xPos + xStep*veci.data[0]);
  x0.data[1] = (float)(xPos + xStep*veci.data[1]);
  x0.data[2] = (float)(xPos + xStep*veci.data[2]);
  x0.data[3] = (float)(xPos + xStep*veci.data[3]);
  float4 y0;
  y0.data[0] = (float)(yPos + yStep*vecj.data[0]);
  y0.data[1] = (float)(yPos + yStep*vecj.data[1]);
  y0.data[2] = (float)(yPos + yStep*vecj.data[2]);
  y0.data[3] = (float)(yPos + yStep*vecj.data[3]);

  float4 x = x0;
  float4 y = y0;

  uint iter = 0;
  float4 tmp;
  int4 stay;
  int4 ccount = make_int4(0, 0, 0, 0);

  float4 savx = x;
  float4 savy = y;

  stay.data[0] = (x.data[0]*x.data[0]+y.data[0]*y.data[0]) <= (float)(4.0f);
  stay.data[1] = (x.data[1]*x.data[1]+y.data[1]*y.data[1]) <= (float)(4.0f);
  stay.data[2] = (x.data[2]*x.data[2]+y.data[2]*y.data[2]) <= (float)(4.0f);
  stay.data[3] = (x.data[3]*x.data[3]+y.data[3]*y.data[3]) <= (float)(4.0f);

  for (iter = 0; (stay.data[0] | stay.data[1] | stay.data[2] | stay.data[3]) && (iter < maxIter);
  iter+=16) {


    x = savx;
    y = savy;

    // Two iterations
    tmp = x*x + x0 - y*y;
    y = 2.0f * x * y + y0;
    x = tmp*tmp + x0 - y*y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x*x + x0 - y*y;
    y = 2.0f * x * y + y0;
    x = tmp*tmp + x0 - y*y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x*x + x0 - y*y;
    y = 2.0f * x * y + y0;
    x = tmp*tmp + x0 - y*y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x*x + x0 - y*y;
    y = 2.0f * x * y + y0;
    x = tmp*tmp + x0 - y*y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x*x + x0 - y*y;
    y = 2.0f * x * y + y0;
    x = tmp*tmp + x0 - y*y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x*x + x0 - y*y;
    y = 2.0f * x * y + y0;
    x = tmp*tmp + x0 - y*y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x*x + x0 - y*y;
    y = 2.0f * x * y + y0;
    x = tmp*tmp + x0 - y*y;
    y = 2.0f * tmp * y + y0;

    stay.data[0] = (x.data[0]*x.data[0]+y.data[0]*y.data[0]) <= (float)(4.0f);
    stay.data[1] = (x.data[1]*x.data[1]+y.data[1]*y.data[1]) <= (float)(4.0f);
    stay.data[2] = (x.data[2]*x.data[2]+y.data[2]*y.data[2]) <= (float)(4.0f);
    stay.data[3] = (x.data[3]*x.data[3]+y.data[3]*y.data[3]) <= (float)(4.0f);

    savx.data[0] = (bool)(stay.data[0] ? x.data[0] : savx.data[0]);
    savx.data[1] = (bool)(stay.data[1] ? x.data[1] : savx.data[1]);
    savx.data[2] = (bool)(stay.data[2] ? x.data[2] : savx.data[2]);
    savx.data[3] = (bool)(stay.data[3] ? x.data[3] : savx.data[3]);

    savy.data[0] = (bool)(stay.data[0] ? y.data[0] : savy.data[0]);
    savy.data[1] = (bool)(stay.data[1] ? y.data[1] : savy.data[1]);
    savy.data[2] = (bool)(stay.data[2] ? y.data[2] : savy.data[2]);
    savy.data[3] = (bool)(stay.data[3] ? y.data[3] : savy.data[3]);

    ccount.data[0] -= stay.data[0]*16;
    ccount.data[1] -= stay.data[1]*16;
    ccount.data[2] -= stay.data[2]*16;
    ccount.data[3] -= stay.data[3]*16;
    }


  // Handle remainder
  if (!(stay.data[0] & stay.data[1] & stay.data[2] & stay.data[3]))
  {
    iter = 16;
    do
    {
      x = savx;
      y = savy;
      stay.x = ((x.data[0]*x.data[0]+y.data[0]*y.data[0]) <= 4.0f) && (ccount.data[0] <  maxIter);
      stay.y = ((x.data[1]*x.data[1]+y.data[1]*y.data[1]) <= 4.0f) && (ccount.data[1] <  maxIter);
      stay.z = ((x.data[2]*x.data[2]+y.data[2]*y.data[2]) <= 4.0f) && (ccount.data[2] <  maxIter);
      stay.w = ((x.data[3]*x.data[3]+y.data[3]*y.data[3]) <= 4.0f) && (ccount.data[3] <  maxIter);
      tmp = x;
      x = x*x + x0 - y*y;
      y = 2.0f*tmp*y + y0;
      ccount.data[0] += stay.data[0];
      ccount.data[1] += stay.data[1];
      ccount.data[2] += stay.data[2];
      ccount.data[3] += stay.data[3];
      iter--;
      savx.data[0] = (stay.data[0] ? x.data[0] : savx.data[0]);
      savx.data[1] = (stay.data[1] ? x.data[1] : savx.data[1]);
      savx.data[2] = (stay.data[2] ? x.data[2] : savx.data[2]);
      savx.data[3] = (stay.data[3] ? x.data[3] : savx.data[3]);
      savy.data[0] = (stay.data[0] ? y.data[0] : savy.data[0]);
      savy.data[1] = (stay.data[1] ? y.data[1] : savy.data[1]);
      savy.data[2] = (stay.data[2] ? y.data[2] : savy.data[2]);
      savy.data[3] = (stay.data[3] ? y.data[3] : savy.data[3]);
    } while ((stay.data[0] | stay.data[1] | stay.data[2] | stay.data[3]) && iter);
  }


  uint4 *vecOut = (uint4 *)out;

  vecOut[tid].data[0] = (uint)(ccount.data[0]);
  vecOut[tid].data[1] = (uint)(ccount.data[1]);
  vecOut[tid].data[2] = (uint)(ccount.data[2]);
  vecOut[tid].data[3] = (uint)(ccount.data[3]);
}


class hipPerfStreamConcurrency {
  public:
  hipPerfStreamConcurrency();
  ~hipPerfStreamConcurrency();

  void setNumKernels(unsigned int num) {
    numKernels = num;
  }
  void setNumStreams(unsigned int num) {
    numStreams = num;
  }
  unsigned int getNumStreams() {
    return numStreams;
  }

  unsigned int getNumKernels() {
    return numKernels;
  }

  void open(int deviceID);
  void run(unsigned int testCase, unsigned int deviceId);
  void close(void);

  private:
  void setData(void *ptr, unsigned int value);
  void checkData(uint *ptr);

  unsigned int numKernels;
  unsigned int numStreams;

  unsigned int width_;
  unsigned int bufSize;
  unsigned int maxIter;
  unsigned int coordIdx;
  unsigned long long totalIters;
  int numCUs;

};


hipPerfStreamConcurrency::hipPerfStreamConcurrency() {}

hipPerfStreamConcurrency::~hipPerfStreamConcurrency() {}

void hipPerfStreamConcurrency::open(int deviceId) {


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

  numCUs = props.multiProcessorCount;
}


void hipPerfStreamConcurrency::close() {
}


void hipPerfStreamConcurrency::run(unsigned int testCase,unsigned int deviceId) {

  int clkFrequency = 0;
  unsigned int numStreams = getNumStreams();
  unsigned int numKernels = getNumKernels();

  HIPCHECK(hipDeviceGetAttribute(&clkFrequency, hipDeviceAttributeClockRate, deviceId));

  clkFrequency =(unsigned int)clkFrequency/1000;

  // Maximum iteration count
  // maxIter = 8388608 * (engine_clock / 1000).serial execution
  maxIter = (unsigned int)(((8388608 * ((float)clkFrequency / 1000)) * numCUs) / 128);
  maxIter = (maxIter + 15) & ~15;

  hipStream_t streams[numStreams];

  uint * hPtr[numKernels];
  uint * dPtr[numKernels];

  // Width is divisible by 4 because the mandelbrot kernel processes 4 pixels at once.
  width_ = 256;

  bufSize = width_ * sizeof(uint);

  // Create streams for concurrency
  for (uint i = 0; i < numStreams; i++) {
    HIPCHECK(hipStreamCreate(&streams[i]));
  }


  // Allocate memory on the host and device
  for (uint i = 0; i < numKernels; i++) {
    HIPCHECK(hipHostMalloc((void **)&hPtr[i], bufSize, hipHostMallocDefault));
    setData(hPtr[i], 0xdeadbeef);
    HIPCHECK(hipMalloc((uint **)&dPtr[i], bufSize))
  }


  // Prepare kernel launch parameters
  int threads = (bufSize/sizeof(uint));
  int threads_per_block  = 64;
  int blocks = (threads/threads_per_block) + (threads % threads_per_block);

  coordIdx = testCase % numCoords;
  float xStep = (float)(coords[coordIdx].width / (double)width_);
  float yStep = (float)(-coords[coordIdx].width / (double)width_);
  float xPos = (float)(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
  float yPos = (float)(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

  // Copy memory asynchronously and concurrently from host to device
  for (uint i = 0; i < numKernels; i++) {
    HIPCHECK(hipMemcpyHtoDAsync(dPtr[i], hPtr[i], bufSize, streams[i % numStreams]));
  }


  // Synchronize to make sure all the copies are completed
  for(uint i = 0; i < numStreams; i++) {
    HIPCHECK(hipStreamSynchronize(streams[i]));
  }

  // Warm-up kernel with lower iteration
  if (testCase == 0) {
    maxIter = 256;
  }

  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();

  for (uint i = 0; i < numKernels; i++) {
    hipLaunchKernelGGL(mandelbrot, dim3(blocks), dim3(threads_per_block), 0, streams[i%numStreams],
                      dPtr[i], width_, xPos, yPos, xStep, yStep, maxIter);
  }


  // Synchronize all the concurrent streans to have completed execution
  for(uint i = 0; i < numStreams; i++) {
    HIPCHECK(hipStreamSynchronize(streams[i]));
  }


  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  // Copy data back from device to the host
  for(uint i = 0; i < numKernels; i++) {
    HIPCHECK(hipMemcpyDtoHAsync(hPtr[i] ,dPtr[i], bufSize, streams[i % numStreams]));
  }


  if (testCase != 0) {
  std::cout <<"Measured time for " << numKernels <<" kernels (s) on " << numStreams <<" stream (s): "
    << all_kernel_time.count() << std::endl;
  }


  unsigned long long expected =
    (unsigned long long)width_ * (unsigned long long)maxIter;

  for(uint i = 0 ; i < numStreams; i++) {
    HIPCHECK(hipStreamDestroy(streams[i]));
  }


  // Free host and device memory
  for (uint i = 0; i < numKernels; i++) {
    HIPCHECK(hipFree(hPtr[i]));
    HIPCHECK(hipFree(dPtr[i]));
  }


}


void hipPerfStreamConcurrency::setData(void *ptr, unsigned int value) {
  unsigned int *ptr2 = (unsigned int *)ptr;
  for (unsigned int i = 0; i < width_ ; i++) {
      ptr2[i] = value;
  }
}


void hipPerfStreamConcurrency::checkData(uint *ptr) {
  totalIters = 0;
  for (unsigned int i = 0; i < width_; i++) {
    totalIters += ptr[i];
  }
}


int main(int argc, char* argv[]) {
  hipPerfStreamConcurrency streamConcurrency;
  int deviceId = 0;

  streamConcurrency.open(deviceId);

  for (unsigned int testCase = 0; testCase < 5; testCase++) {


  switch (testCase) {


  case 0:
    // Warm-up kernel
    streamConcurrency.setNumStreams(1);
    streamConcurrency.setNumKernels(1);
    break;

  case 1:
  // default stream executes serially
  streamConcurrency.setNumStreams(1);
  streamConcurrency.setNumKernels(1);
  break;

  case 2:
    // 2-way concurrency
    streamConcurrency.setNumStreams(2);
    streamConcurrency.setNumKernels(2);
    break;

  case 3:
    // 4-way concurrency
    streamConcurrency.setNumStreams(4);
    streamConcurrency.setNumKernels(4);
    break;

  case 4:
    streamConcurrency.setNumStreams(2);
    streamConcurrency.setNumKernels(4);
    break;

  case 5:
    break;

  default:
    break;
  }
  streamConcurrency.run(testCase, deviceId);

  }


  passed();
}
