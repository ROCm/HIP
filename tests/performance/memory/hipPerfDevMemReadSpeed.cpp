/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

#define arraySize 16

typedef struct d_uint16 {
  uint data[arraySize];
} d_uint16;

__global__ void read_kernel(d_uint16 *src, ulong N, uint *dst) {

  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x ;

  uint tmp = 0;
  for (size_t i = idx; i < N; i += stride) {
    for (size_t j = 0; j < arraySize; j++) {
      tmp += src[i].data[j];
    }
  }

  atomicAdd(dst, tmp);
}

int main(int argc, char* argv[]) {
  d_uint16 *dSrc;
  d_uint16 *hSrc;
  uint *dDst;
  uint *hDst;
  hipStream_t stream;
  ulong N = 4 * 1024 * 1024;
  uint nBytes = N * sizeof(d_uint16);

  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
      cout << "info: didn't find any GPU! skipping the test!\n";
      passed();
      return 0;
  }

  static int device = 0;
  HIPCHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device));
  cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name <<
      " with " << props.multiProcessorCount << " CUs" << endl;

  const unsigned threadsPerBlock = 64;
  const unsigned blocks = props.multiProcessorCount * 4;

  uint inputData = 0x1;
  int nIter = 1000;

  hSrc =  new d_uint16[nBytes];
  HIPCHECK(hSrc == 0 ? hipErrorOutOfMemory : hipSuccess);
  hDst =  new uint;
  hDst[0] = 0;
  HIPCHECK(hDst == 0 ? hipErrorOutOfMemory : hipSuccess);
  for (size_t i = 0; i < N; i++) {
    for (int j = 0; j < arraySize; j++) {
      hSrc[i].data[j] = inputData;
    }
  }

  HIPCHECK(hipMalloc(&dSrc, nBytes));
  HIPCHECK(hipMalloc(&dDst, sizeof(uint)));

  HIPCHECK(hipStreamCreate(&stream));

  HIPCHECK(hipMemcpy(dSrc, hSrc, nBytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(dDst, hDst, sizeof(uint), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dSrc, N, dDst);
  HIPCHECK(hipMemcpy(hDst, dDst, sizeof(uint), hipMemcpyDeviceToHost));
  hipDeviceSynchronize();

  if (hDst[0] != (nBytes / sizeof(uint))) {
    cout << "info: Data validation failed for warm up run!" << endl;
    cout << "info: expected " << nBytes / sizeof(uint) << " got " << hDst[0] << endl;
    HIPCHECK(hipErrorUnknown);
  }

  // measure performance based on host time
  auto all_start = chrono::steady_clock::now();

  for(int i = 0; i < nIter; i++) {
    hipLaunchKernelGGL(read_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dSrc, N, dDst);
  }
  hipDeviceSynchronize();

  auto all_end = chrono::steady_clock::now();
  chrono::duration<double> all_kernel_time = all_end - all_start;

  // read speed in GB/s
  double perf = ((double)nBytes * nIter * (double)(1e-09)) / all_kernel_time.count();

  cout << "info: average read speed of " << perf << " GB/s " << "achieved for memory size of " <<
      nBytes / (1024 * 1024) << " MB" << endl;

  delete [] hSrc;
  delete hDst;
  hipFree(dSrc);
  hipFree(dDst);
  HIPCHECK(hipStreamDestroy(stream));

  passed();
}
