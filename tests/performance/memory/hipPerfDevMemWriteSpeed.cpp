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

using namespace std;

#define arraySize 16

typedef struct d_uint16 {
  uint data[arraySize];
} d_uint16;

__global__ void write_kernel(d_uint16 *dst, ulong N, d_uint16 pval) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride) {
      dst[i] = pval;
    }
};

int main(int argc, char* argv[]) {
  d_uint16 *dDst;
  d_uint16 *hDst;
  hipStream_t stream;
  ulong N = 4 * 1024 * 1024;
  uint nBytes = N * sizeof(d_uint16);
  d_uint16 pval;

  for (int i = 0; i < arraySize; i++) {
    pval.data[i] = 0xabababab;
  }

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

  size_t threadsPerBlock = 64;
  size_t blocks = props.multiProcessorCount * 4;

  uint inputData = 0xabababab;
  int nIter = 1000;

  hDst =  new d_uint16[nBytes];
  HIPCHECK(hDst == 0 ? hipErrorOutOfMemory : hipSuccess);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < arraySize; j++) {
      hDst[i].data[j] = 0;
    }
  }

  HIPCHECK(hipMalloc(&dDst, nBytes));

  HIPCHECK(hipStreamCreate(&stream));

  hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst, N, pval);
  HIPCHECK(hipMemcpy(hDst, dDst, nBytes , hipMemcpyDeviceToHost));
  hipDeviceSynchronize();

  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < arraySize; j++) {
      if (hDst[i].data[j] != inputData) {
        cout << "info: Data validation failed for warm up run! " << endl;
        cout << "at index i: " << i << " element j: " << j << endl;
        cout << hex << "expected 0x" << inputData << " but got 0x" << hDst[i].data[j] << endl;
        HIPCHECK(hipErrorUnknown);
      }
    }
  }

  auto all_start = chrono::steady_clock::now();
  for(int i = 0; i < nIter; i++) {
    hipLaunchKernelGGL(write_kernel, dim3(blocks), dim3(threadsPerBlock), 0, stream, dDst, N, pval);
  }
  hipDeviceSynchronize();
  auto all_end = chrono::steady_clock::now();
  chrono::duration<double> all_kernel_time = all_end - all_start;

  // read speed in GB/s
  double perf = ((double)nBytes * nIter * (double)(1e-09)) / all_kernel_time.count();

  cout << "info: average write speed of " << perf << " GB/s " << "achieved for memory size of " <<
      nBytes / (1024 * 1024) << " MB" << endl;


  delete [] hDst;
  hipFree(dDst);
  HIPCHECK(hipStreamDestroy(stream));

  passed();
}
