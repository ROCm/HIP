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
#include <hip/math_functions.h>
#include <vector>
#include <string>
#include <map>

typedef struct {
  double x;
  double y;
  double width;
} coordRec;

coordRec coords[] = {
    {0.0, 0.0, 4.0},                                     // Whole set
    {0.0, 0.0, 0.00001},                                 // All black
    {-0.0180789661868, 0.6424294066162, 0.00003824140},  // Hit detail
};

static unsigned int numCoords = sizeof(coords) / sizeof(coordRec);

template <typename T>
__global__ void float_mad_kernel(uint *out, uint width, T xPos,  T yPos, T xStep, T yStep,
                                  uint maxIter) {

#pragma FP_CONTRACT ON
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

template <typename T>
__global__ void float_mandel_unroll_kernel(uint *out, uint width, T xPos,
    T yPos, T xStep, T yStep, uint maxIter) {

#pragma FP_CONTRACT ON
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  int i = tid % width;
  int j = tid / width;
  float x0 = (float)(xPos + xStep*(float)i);
  float y0 = (float)(yPos + yStep*(float)j);

  float x = x0;
  float y = y0;

#define FAST
  uint iter = 0;
  float tmp;
  int stay;
  int ccount = 0;
  stay = (x*x+y*y) <= 4.0;
  float savx = x;
  float savy = y;
#ifdef FAST
  for (iter = 0; (iter < maxIter); iter+=16) {
#else
  for (iter = 0; stay && (iter < maxIter); iter+=16) {
#endif
    x = savx;
    y = savy;

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =  fma(2.0f*x,y,y0);
    x =  fma(-y,y, fma(tmp,tmp,x0));
    y =  fma(2.0f*tmp,y,y0);

    stay = (x*x+y*y) <= 4.0;
    savx = (stay ? x : savx);
    savy = (stay ? y : savy);
    ccount += stay*16;
#ifdef FAST
    if (!stay)
      break;
#endif
  }
  // Handle remainder
  if (!stay) {
    iter = 16;
    do {
      x = savx;
      y = savy;
      stay = ((x*x+y*y) <= 4.0) && (ccount < maxIter);
      tmp = x;
      x =  fma(-y,y, fma(x,x,x0));
      y =  fma(2.0f*tmp,y,y0);
      ccount += stay;
      iter--;
      savx = (stay ? x : savx);
      savy = (stay ? y : savy);
    } while (stay && iter);
  }


  out[tid] = (uint)ccount;

};


template <typename T>
__global__ void double_mad_kernel(uint *out, uint width, T xPos,  T yPos, T xStep, T yStep,
                                   uint maxIter) {

#pragma FP_CONTRACT ON
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  int i = tid % width;
  int j = tid / width;
  double x0 = (double)(xPos + xStep*i);
  double y0 = (double)(yPos + yStep*j);

  double x = x0;
  double y = y0;

  uint iter = 0;
  double tmp;
  for (iter = 0; (x*x + y*y <= 4.0f) && (iter < maxIter); iter++) {
    tmp = x;
    x = fma(-y,y,fma(x,x,x0));
    y = fma(2.0f*tmp,y,y0);
  }
  out[tid] = iter;
};


template <typename T>
__global__ void double_mandel_unroll_kernel(uint *out, uint width, T xPos,
                  T yPos, T xStep, T yStep, uint maxIter) {

#pragma FP_CONTRACT ON
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);

  int i = tid % width;
  int j = tid / width;
  double x0 = (double)(xPos + xStep*(double)i);
  double y0 = (double)(yPos + yStep*(double)j);

  double x = x0;
  double y = y0;

#define FAST
  uint iter = 0;
  double tmp;
  int stay;
  int ccount = 0;
  stay = (x*x+y*y) <= 4.0;
  double savx = x;
  double savy = y;
#ifdef FAST
  for (iter = 0; (iter < maxIter); iter+=16)
#else
  for (iter = 0; stay && (iter < maxIter); iter+=16)
#endif
  {
    x = savx;
    y = savy;

    // Two iterations
    tmp = fma(-y,y, fma(x,x,x0));
    y =   fma(2.0f*x,y,y0);
    x =   fma(-y,y, fma(tmp,tmp,x0));
    y =   fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp = fma(-y,y, fma(x,x,x0));
    y =   fma(2.0f*x,y,y0);
    x =   fma(-y,y, fma(tmp,tmp,x0));
    y =   fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp = fma(-y,y, fma(x,x,x0));
    y =   fma(2.0f*x,y,y0);
    x =   fma(-y,y, fma(tmp,tmp,x0));
    y =   fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =    fma(2.0f*x,y,y0);
    x =    fma(-y,y, fma(tmp,tmp,x0));
    y =    fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =    fma(2.0f*x,y,y0);
    x =    fma(-y,y, fma(tmp,tmp,x0));
    y =    fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =    fma(2.0f*x,y,y0);
    x =    fma(-y,y, fma(tmp,tmp,x0));
    y =    fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =    fma(2.0f*x,y,y0);
    x =    fma(-y,y, fma(tmp,tmp,x0));
    y =    fma(2.0f*tmp,y,y0);

    // Two iterations
    tmp =  fma(-y,y, fma(x,x,x0));
    y =    fma(2.0f*x,y,y0);
    x =    fma(-y,y, fma(tmp,tmp,x0));
    y =    fma(2.0f*tmp,y,y0);

    stay = (x*x+y*y) <= 4.0;
    savx = (stay ? x : savx);
    savy = (stay ? y : savy);
    ccount += stay*16;
#ifdef FAST
    if (!stay)
      break;
#endif
    }
  // Handle remainder
    if (!stay) {
      iter = 16;
      do {
        x = savx;
        y = savy;
        stay = ((x*x+y*y) <= 4.0) && (ccount < maxIter);
        tmp = x;
        x =  fma(-y,y, fma(x,x,x0));
        y =  fma(2.0f*tmp,y,y0);
        ccount += stay;
        iter--;
        savx = (stay ? x : savx);
        savy = (stay ? y : savy);
      }
      while (stay && iter);

    }
    out[tid] = (uint)ccount;
};

static const unsigned int FMA_EXPECTEDVALUES_INDEX = 15;

// Expected results for each kernel run at each coord
unsigned long long expectedIters[] = {
    203277748ull,  2147483648ull, 120254651ull,  203277748ull,  2147483648ull,
    120254651ull,  203277748ull,  2147483648ull, 120254651ull,  203315114ull,
    2147483648ull, 120042599ull,  203315114ull,  2147483648ull, 120042599ull,
    203280620ull,  2147483648ull, 120485704ull,  203280620ull,  2147483648ull,
    120485704ull,  203280620ull,  2147483648ull, 120485704ull,  203315114ull,
    2147483648ull, 120042599ull,  203315114ull,  2147483648ull, 120042599ull};

class hipPerfMandelBrot {
  public:
  hipPerfMandelBrot();
  ~hipPerfMandelBrot();

  void setNumKernels(unsigned int num) {
    numKernels = num;
  }

  unsigned int getNumKernels() {
    return numKernels;
  }

  void setNumStreams(unsigned int num) {
    numStreams = num;
  }
  unsigned int getNumStreams() {
    return numStreams;
  }

  void open(int deviceID);
  void run(unsigned int testCase, unsigned int deviceId);
  void printResults(void);

  // array of funtion pointers
  typedef void (hipPerfMandelBrot::*funPtr)(uint *out, uint width, float xPos,  float yPos,
                 float xStep, float yStep, uint maxIter,  hipStream_t* streams, int blocks,
                 int threads_per_block, int kernelCnt);

  // Wrappers
  void float_mad(uint *out, uint width, float xPos,  float yPos,
                  float xStep, float yStep, uint maxIter, hipStream_t* streams,
                  int blocks, int threads_per_block, int kernelCnt);

  void float_mandel_unroll(uint *out, uint width, float xPos,  float yPos,
                            float xStep, float yStep, uint maxIter, hipStream_t* streams,
                            int blocks, int threads_per_block, int kernelCnt);

  void double_mad(uint *out, uint width, float xPos,  float yPos, float xStep,
                   float yStep, uint maxIter, hipStream_t* streams, int blocks,
                   int threads_per_block, int kernelCnt);

  void double_mandel_unroll(uint *out, uint width, float xPos,  float yPos, float xStep,
                             float yStep, uint maxIter, hipStream_t* streams, int blocks,
                             int threads_per_block, int kernelCnt);

  hipStream_t streams[2];

  private:
  void setData(void *ptr, unsigned int value);
  void checkData(uint *ptr);

  unsigned int numKernels;
  unsigned int numStreams;

  std::map<std::string, std::vector<double>> results;
  unsigned int width_;
  unsigned int bufSize;
  unsigned int maxIter;
  unsigned int coordIdx;
  volatile unsigned long long totalIters = 0;
  int numCUs;
  static const unsigned int numLoops = 10;
};


hipPerfMandelBrot::hipPerfMandelBrot() {}

hipPerfMandelBrot::~hipPerfMandelBrot() {}

void hipPerfMandelBrot::open(int deviceId) {


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
    << " with " << props.multiProcessorCount << " CUs" << " and device id: " << deviceId
    << std::endl;

  numCUs = props.multiProcessorCount;
}


void hipPerfMandelBrot::printResults() {

  int numkernels = getNumKernels();
  int numStreams = getNumStreams();

  std::cout << "\n" <<"Measured perf for kernels in GFLOPS on "
            << numStreams << " streams (s)" <<  std::endl;

  std::map<std::string, std::vector<double>>:: iterator itr;
  for (itr = results.begin(); itr != results.end(); itr++) {
          std::cout << "\n" << std::setw(20) << itr->first << " ";
          for(auto i : results[itr->first]) {
            std::cout << std::setw(10) << i << " ";
            }
     }
  results.clear();

  std::cout << std::endl;
}


// Wrappers for the kernel launches
void hipPerfMandelBrot::float_mad(uint *out, uint width, float xPos,  float yPos, float xStep,
                                   float yStep, uint maxIter, hipStream_t* streams,
                                   int blocks, int threads_per_block, int kernelCnt) {

  int streamCnt = getNumStreams();
  hipLaunchKernelGGL(float_mad_kernel<float>, dim3(blocks), dim3(threads_per_block), 0,
                      streams[kernelCnt % streamCnt], out, width_, xPos, yPos, xStep, yStep,
                      maxIter);


}


void hipPerfMandelBrot::float_mandel_unroll(uint *out, uint width, float xPos,  float yPos,
                             float xStep, float yStep, uint maxIter, hipStream_t * streams,
                             int blocks, int threads_per_block, int kernelCnt) {

  int streamCnt = getNumStreams();
  hipLaunchKernelGGL(float_mandel_unroll_kernel<float>, dim3(blocks), dim3(threads_per_block), 0,
                  streams[kernelCnt % streamCnt], out, width_, xPos, yPos, xStep, yStep, maxIter);

}


void hipPerfMandelBrot::double_mad(uint *out, uint width, float xPos,  float yPos,
                               float xStep, float yStep, uint maxIter, hipStream_t * streams,
                               int blocks, int threads_per_block, int kernelCnt) {

  int streamCnt = getNumStreams();
  hipLaunchKernelGGL(double_mad_kernel<double>, dim3(blocks), dim3(threads_per_block), 0,
                  streams[kernelCnt % streamCnt], out, width_, xPos, yPos, xStep, yStep, maxIter);

}


void hipPerfMandelBrot::double_mandel_unroll(uint *out, uint width, float xPos,  float yPos,
                              float xStep, float yStep, uint maxIter, hipStream_t * streams,
                              int blocks, int threads_per_block, int kernelCnt) {

  int streamCnt = getNumStreams();
  hipLaunchKernelGGL(float_mandel_unroll_kernel<double>, dim3(blocks), dim3(threads_per_block), 0,
                  streams[kernelCnt % streamCnt], out, width_, xPos, yPos, xStep, yStep, maxIter);

}


void hipPerfMandelBrot::run(unsigned int testCase,unsigned int deviceId) {

  unsigned int numStreams = getNumStreams();

  funPtr p[] = {&hipPerfMandelBrot::float_mad, &hipPerfMandelBrot::float_mandel_unroll,
               &hipPerfMandelBrot::double_mad, &hipPerfMandelBrot::double_mandel_unroll};

  // Maximum iteration count
  maxIter = 32768;

  uint * hPtr[numKernels];
  uint * dPtr[numKernels];

  // Width is divisible by 4 because the mandelbrot kernel processes 4 pixels at once.
  width_ = 256;

  bufSize = width_  * width_ * sizeof(uint);

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

  float xStep = (float)(coords[coordIdx].width / (double)width_);
  float yStep = (float)(-coords[coordIdx].width / (double)width_);
  float xPos = (float)(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
  float yPos = (float)(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

  // Copy memory asynchronously and concurrently from host to device
  for (uint i = 0; i < numKernels; i++) {
    HIPCHECK(hipMemcpy(dPtr[i], hPtr[i], bufSize, hipMemcpyHostToDevice));
  }

  // Synchronize to make sure all the copies are completed
  HIPCHECK(hipStreamSynchronize(0));

  int kernelIdx;
  if(testCase == 0 || testCase == 5 || testCase == 10) {
    kernelIdx = 0;
  }

  else if(testCase == 1 || testCase == 6 || testCase == 11) {
    kernelIdx = 1;
  }
  else if(testCase == 2 || testCase == 7 || testCase == 12) {
    kernelIdx = 2;
  }
  else if(testCase == 3 || testCase == 8 || testCase == 13){
    kernelIdx = 3;
  }


  double totalTime = 0.0;

  for (unsigned int k = 0; k < numLoops; k++) {

  coordIdx = testCase % numCoords;

  if ((testCase == 0 || testCase == 1 || testCase == 2 ||
                  testCase == 5 || testCase == 6 || testCase == 7 ||
                  testCase == 10 || testCase == 11 || testCase == 12)) {
  float xStep = (float)(coords[coordIdx].width / (double)width_);
  float yStep = (float)(-coords[coordIdx].width / (double)width_);
  float xPos = (float)(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
  float yPos = (float)(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();

  for (uint i = 0; i < numKernels; i++) {
    (this->*p[kernelIdx])(dPtr[i], width_, xPos, yPos, xStep, yStep, maxIter, streams, blocks,
                           threads_per_block, i);
  }


  // Synchronize all the concurrent streams to have completed execution
  HIPCHECK(hipStreamSynchronize(0));

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;
  totalTime += all_kernel_time.count();

  }


  else {
  double xStep = coords[coordIdx].width / (double)width_;
  double yStep = -coords[coordIdx].width / (double)width_;
  double xPos = coords[coordIdx].x - 0.5 * coords[coordIdx].width;
  double yPos = coords[coordIdx].y + 0.5 * coords[coordIdx].width;

  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();

  for (uint i = 0; i < numKernels; i++) {
  (this->*p[kernelIdx])(dPtr[i], width_, xPos, yPos, xStep, yStep, maxIter, streams, blocks,
                         threads_per_block, i);
  }


  // Synchronize all the concurrent streams to have completed execution
  HIPCHECK(hipStreamSynchronize(0));

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;
  totalTime += all_kernel_time.count();
  }


  }

  // Copy data back from device to the host
  for(uint i = 0; i < numKernels; i++) {
    HIPCHECK(hipMemcpy(hPtr[i] ,dPtr[i], bufSize, hipMemcpyDeviceToHost));
  }


  for(uint i = 0; i < numKernels; i++) {
  checkData(hPtr[i]);

  int j =0;
  while((totalIters != expectedIters[j] && totalIters > expectedIters[j]) && j < 30) {
          j++;
  }

  if(j==30) {
    std::cout << "Incorrect iteration count detected. ";
  }

  }


  // Compute GFLOPS.  There are 7 FLOPs per iteration
  double perf = ((double)(totalIters*numKernels) * 7 * (double)(1e-09)) /
                (totalTime / (double)numLoops);


  std::vector<std::string> kernelName = {"float", "float_unroll",
                      "double", "double_unroll"};

  // Print results except for Warm-up kernel
  if(testCase!=100) {
  results[kernelName[testCase % 4]].push_back(perf);
 }


  for(uint i = 0 ; i < numStreams; i++) {
    HIPCHECK(hipStreamDestroy(streams[i]));
  }


  // Free host and device memory
  for (uint i = 0; i < numKernels; i++) {
    HIPCHECK(hipFree(hPtr[i]));
    HIPCHECK(hipFree(dPtr[i]));
  }


}


void hipPerfMandelBrot::setData(void *ptr, unsigned int value) {
  unsigned int *ptr2 = (unsigned int *)ptr;
  for (unsigned int i = 0; i < width_ * width_; i++) {
      ptr2[i] = value;
  }
}


void hipPerfMandelBrot::checkData(uint *ptr) {
  totalIters = 0;
  for (unsigned int i = 0; i < width_ * width_; i++) {
    totalIters += ptr[i];
  }
}


int main(int argc, char* argv[]) {
  hipPerfMandelBrot mandelbrotCompute;
  int deviceId = 0;

  mandelbrotCompute.open(deviceId);

  for (unsigned int testCase = 0; testCase < 3; testCase++) {


  switch (testCase) {


  case 0: {
    // Warmup-kernel - default stream executes serially
    mandelbrotCompute.setNumStreams(1);
    mandelbrotCompute.setNumKernels(1);
    mandelbrotCompute.run(100/*Random number*/, deviceId);
    break;
    }


  case 1: {
    // run all - sync
    int i = 0;
    do {
    mandelbrotCompute.setNumStreams(1);
    mandelbrotCompute.setNumKernels(1);
    mandelbrotCompute.run(i, deviceId);
    i++;
    }while(i < 12);
    mandelbrotCompute.printResults();

    break;
  }


  case 2: {
    // run all - async
    int i = 0;
    do {
    mandelbrotCompute.setNumStreams(2);
    mandelbrotCompute.setNumKernels(2);
    mandelbrotCompute.run(i, deviceId);
    i++;
    }while(i < 12);
    mandelbrotCompute.printResults();

    break;

  }


  default: {
    break;
  }


  }



  }


  passed();
}
