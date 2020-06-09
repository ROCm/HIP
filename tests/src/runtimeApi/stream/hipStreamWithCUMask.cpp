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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include <iostream>
#include <vector>
#include <chrono>
#include "test_common.h"

using namespace std;

__global__ void vector_square(float *C_d, float *A_d, size_t N) {
    size_t idx    = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i = idx; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

int main(int argc, char* argv[]) {
  constexpr uint32_t numPartition = 4;
  float *dA[numPartition], *dC[numPartition];
  float *hA, *hC;
  size_t N = 25 * 1024 * 1024;
  size_t Nbytes = N * sizeof(float);
  vector<hipStream_t> streams(numPartition);
  vector<vector<uint32_t>> cuMasks(numPartition);
  stringstream ss[numPartition];

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

  hA =  new float[Nbytes];
  HIPCHECK(hA == 0 ? hipErrorOutOfMemory : hipSuccess);
  hC =  new float[Nbytes];
  HIPCHECK(hC == 0 ? hipErrorOutOfMemory : hipSuccess);
  for (size_t i = 0; i < N; i++) {
    hA[i] = 1.618f + i;
  }

  for (int np = 0; np < numPartition; np++) {

    HIPCHECK(hipMalloc(&dA[np], Nbytes));
    HIPCHECK(hipMalloc(&dC[np], Nbytes));

    // make unique CU masks in the multiple of dwords for each stream
    uint32_t temp = 0;
    uint32_t bit_index = np;
    for (int i = np; i < props.multiProcessorCount; i = i + 4) {
      temp |= 1UL << bit_index;
      if (bit_index >= 32) {
        cuMasks[np].push_back(temp);
        temp = 0;
        bit_index = np;
        temp |= 1UL << bit_index;
      }
      bit_index += 4;
    }
    if (bit_index != 0) {
      cuMasks[np].push_back(temp);
    }

    HIPCHECK(hipExtStreamCreateWithCUMask(&streams[np], cuMasks[np].size(), cuMasks[np].data()));

    HIPCHECK(hipMemcpy(dA[np], hA, Nbytes, hipMemcpyHostToDevice));

    ss[np] << std::hex;
    for (int i = cuMasks[np].size() - 1; i >= 0; i--) {
      ss[np] << cuMasks[np][i];
    }
  }

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;

  auto single_start = chrono::steady_clock::now();
  cout << "info: launch 'vector_square' kernel on one stream " << streams[0] << " with CU mask: 0x" << ss[0].str().c_str() << endl;

  hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0, streams[0], dC[0], dA[0], N);
  hipDeviceSynchronize();

  auto single_end = chrono::steady_clock::now();
  chrono::duration<double> single_kernel_time = single_end - single_start;

  HIPCHECK(hipMemcpy(hC, dC[0], Nbytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    if (hC[i] != hA[i] * hA[i]) {
      cout << "info: validation failed for kernel launched at stream" << streams[0] << endl;
      HIPCHECK(hipErrorUnknown);
    }
  }

  cout << "info: launch 'vector_square' kernel on " << numPartition << " streams:" << endl;
  auto all_start = chrono::steady_clock::now();
  for (int np = 0; np < numPartition; np++) {
    cout << "info: launch 'vector_square' kernel on the stream " << streams[np] << " with CU mask: 0x" << ss[np].str().c_str() << endl;
    hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0,
        streams[np], dC[np], dA[np], N);
  }
  hipDeviceSynchronize();

  auto all_end = chrono::steady_clock::now();
  chrono::duration<double> all_kernel_time = all_end - all_start;

  for (int np = 0; np < numPartition; np++) {
    HIPCHECK(hipMemcpy(hC, dC[np], Nbytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < N; i++) {
      if (hC[i] != hA[i] * hA[i]) {
        cout << "info: validation failed for kernel launched at stream" << streams[np] << endl;
        HIPCHECK(hipErrorUnknown);
      }
    }
  }

  cout << "info: kernel launched on one stream took: " << single_kernel_time.count() << " seconds" << endl;
  cout << "info: kernels launched on " << numPartition << " streams took: " << all_kernel_time.count() << " seconds" << endl;
  cout << "info: launching kernels on " << numPartition << " streams asynchronously is " << single_kernel_time.count() / (all_kernel_time.count() / numPartition)
      << " times faster per stream than launching on one stream alone" << endl;

  delete [] hA;
  delete [] hC;
  for (int np = 0; np < numPartition; np++) {
    hipFree(dC[np]);
    hipFree(dA[np]);
    HIPCHECK(hipStreamDestroy(streams[np]));
  }

  passed();
}
