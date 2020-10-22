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
#include <vector>

#define DOT_DIM 256

using namespace std;

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void vectors_not_equal(int n,
                                 const double* __restrict__ x,
                                 const double* __restrict__ y,
                                 double* __restrict__ workspace) {
  int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  double sum = 0.0;
  for(int idx = gid; idx < n; idx += hipGridDim_x * hipBlockDim_x) {
    sum = fma(y[idx], x[idx], sum);
  }

  __shared__ double sdata[BLOCKSIZE];
  sdata[hipThreadIdx_x] = sum;

  __syncthreads();

  if(hipThreadIdx_x < 128) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 128];
  }
  __syncthreads();

  if(hipThreadIdx_x < 64){
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 64];
  }
  __syncthreads();

  if(hipThreadIdx_x < 32){
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 32];
  }
  __syncthreads();

  if(hipThreadIdx_x < 16) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 16];
  }
  __syncthreads();

  if(hipThreadIdx_x < 8) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
  }
  __syncthreads();

  if(hipThreadIdx_x < 4) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
  }
  __syncthreads();

  if(hipThreadIdx_x < 2) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
  }
  __syncthreads();

  if(hipThreadIdx_x < 1) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 1];
  }

  if(hipThreadIdx_x == 0) {
    workspace[hipBlockIdx_x] = sdata[0];
  }

}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void vectors_equal(int n, const double* __restrict__ x,
                                  double* __restrict__ workspace) {
  int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  double sum = 0.0;
  for(int idx = gid; idx < n; idx += hipGridDim_x * hipBlockDim_x) {
    sum = fma(x[idx], x[idx], sum);
  }

  __shared__ double sdata[BLOCKSIZE];
  sdata[hipThreadIdx_x] = sum;

  __syncthreads();

  if(hipThreadIdx_x < 128) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 128];
  }
  __syncthreads();

  if(hipThreadIdx_x < 64) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 64];
  }
  __syncthreads();

  if(hipThreadIdx_x < 32) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 32];
  }
  __syncthreads();

  if(hipThreadIdx_x < 16) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 16];
  }
  __syncthreads();

  if(hipThreadIdx_x < 8) {
          sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
  }
  __syncthreads();

  if(hipThreadIdx_x < 4) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
  }
  __syncthreads();

  if(hipThreadIdx_x < 2) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
  }
  __syncthreads();

  if(hipThreadIdx_x < 1) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 1];
  }

  if(hipThreadIdx_x == 0) {
    workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE)
__global__ void dot_reduction(double* __restrict__ workspace) {

  __shared__ double sdata[BLOCKSIZE];

  sdata[hipThreadIdx_x] = workspace[hipThreadIdx_x];

  __syncthreads();

  if(hipThreadIdx_x < 128) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 128];
  }
  __syncthreads();

  if(hipThreadIdx_x < 64) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 64];
  }
  __syncthreads();

  if(hipThreadIdx_x < 32) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 32];
  }
  __syncthreads();

  if(hipThreadIdx_x < 16) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 16];
  }
  __syncthreads();

  if(hipThreadIdx_x < 8) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
  }
  __syncthreads();

  if(hipThreadIdx_x < 4) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
  } __syncthreads();

  if(hipThreadIdx_x < 2) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
  }
  __syncthreads();

  if(hipThreadIdx_x < 1) {
    sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 1];
  }

  if(hipThreadIdx_x == 0) {
    workspace[0] = sdata[0];
  }

}

void computeDotProduct(int n, const double* x, const double* y, double& result,
                      double* workspace)
{
  dim3 blocks(DOT_DIM);
  dim3 threadsPerBlock(DOT_DIM);

  if(x != y) {
    hipLaunchKernelGGL(vectors_not_equal<DOT_DIM>, blocks, threadsPerBlock, 0, 0, n, x, y,
                       workspace);
  }
  else {
    hipLaunchKernelGGL(vectors_equal<DOT_DIM>, blocks, threadsPerBlock, 0, 0, n, x, workspace);
  }

  // Part 2 of dot product computation
  hipLaunchKernelGGL(dot_reduction<DOT_DIM>, dim3(1), threadsPerBlock, 0, 0, workspace);

  // Copy the final dot product result back from the device
  HIPCHECK(hipMemcpy(&result, workspace, sizeof(double), hipMemcpyDeviceToHost));

  return;
}

int main(int argc, char* argv[]) {

  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));

  if (nGpu < 1) {
  std::cout << "info: didn't find any GPU! skipping the test!\n";
  passed();
  return 0;
  }
  hipDeviceProp_t props = {0};
  props = {0};
  HIPCHECK(hipSetDevice(p_gpuDevice));
  HIPCHECK(hipGetDeviceProperties(&props, p_gpuDevice));
  std::cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name
    << " with " << props.multiProcessorCount << " CUs" << " and device id: " << p_gpuDevice
    << std::endl;

  int nx, ny, nz;

  for (unsigned int testCase = 0; testCase < 3; testCase++) {

  vector<int> vectorSize = {200, 300, 50};
  switch(testCase) {

  case 0:
  nx = vectorSize[0];
  ny = vectorSize[0];
  nz = vectorSize[0];
  break;

  case 1:
  nx = vectorSize[1];
  ny = vectorSize[1];
  nz = vectorSize[1];
  break;

  case 2:
  nx = vectorSize[0];
  ny = vectorSize[1];
  nz = vectorSize[2];
  break;

  default:
    break;

  }

  int trials = 200;

  int size = nx * ny * nz;

  vector<double> hx(size);
  vector<double> hy(size);
  double hresult_xy = 0.0;
  double hresult_xx = 0.0;

  srand(time(NULL));

  for(int i = 0; i < size; ++i) {
    hx[i] = 2.0 * (double)rand() / (double)RAND_MAX - 1.0;
    hy[i] = 2.0 * (double)rand() / (double)RAND_MAX - 1.0;

    hresult_xy += hx[i] * hy[i];
    hresult_xx += hx[i] * hx[i];
  }

  double* dx;
  double* dy;
  double* workspace;
  double  dresult;

  HIPCHECK(hipMalloc((void**)&dx, sizeof(double) * size));
  HIPCHECK(hipMalloc((void**)&dy, sizeof(double) * size));
  HIPCHECK(hipMalloc((void**)&workspace, sizeof(double) * DOT_DIM));

  HIPCHECK(hipMemcpy(dx, hx.data(), sizeof(double) * size, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(dy, hy.data(), sizeof(double) * size, hipMemcpyHostToDevice));

  // Warm up
  computeDotProduct(size, dx, dy, dresult, workspace);
  computeDotProduct(size, dx, dy, dresult, workspace);
  computeDotProduct(size, dx, dy, dresult, workspace);

  // Timed run for <x,y>
  HIPCHECK(hipDeviceSynchronize());
  auto all_start = std::chrono::steady_clock::now();

  for(int i = 0; i < trials; ++i) {
    computeDotProduct(size, dx, dy, dresult, workspace);
  }

  float time = 0;
  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;
  time = all_kernel_time.count();

  time /= trials;

  double bw = sizeof(double) * size * 2.0 / 1e9;
  double gf = 2.0 * size / 1e9;

  cout << "\nVector Size: " << size << "\n[ddot] <x,y> " << time << "msec ;" << bw/ (time / 1e3) << " GByte/s ;"
       << gf/(time / 1e3) << " GFlop/s" << endl;

  // Verify the device kernel results comparing it with the host results
  if(std::abs(dresult - hresult_xy) > std::max(dresult * 1e-10, 1e-8)) {
    cerr << " Device results inconsistent with host results. "
         << " Host result: " << hresult_xy
         << " Device result: " << dresult;
  }

  // Warm up
  computeDotProduct(size, dx, dx, dresult, workspace);
  computeDotProduct(size, dx, dx, dresult, workspace);
  computeDotProduct(size, dx, dx, dresult, workspace);

  // Timed run for <x,x>
  HIPCHECK(hipDeviceSynchronize());
  all_start = std::chrono::steady_clock::now();

  for(int i = 0; i < trials; ++i) {
    computeDotProduct(size, dx, dx, dresult, workspace);
  }

  all_end = std::chrono::steady_clock::now();
  all_kernel_time = all_end - all_start;
  time = all_kernel_time.count();

  time /= trials;
  bw = sizeof(double) * size / 1e9;

  cout << "[ddot] <x,y> " << time << "msec ;" << bw/ (time / 1e3) << " GByte/s ;"
       << gf/(time / 1e3) << " GFlop/s" << endl;

  // Verify the device kernel results comparing it with the host results
  if(abs(dresult - hresult_xx) > max(dresult * 1e-10, 1e-8)) {
    cerr << " Device results inconsistent with host results"
         << " Host result: " << hresult_xy
         << " Device result: " << dresult;
  }

  HIPCHECK(hipFree(dx));
  HIPCHECK(hipFree(dy));
  HIPCHECK(hipFree(workspace));

  }

  return 0;
}
