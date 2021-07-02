/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>


__global__ void warpvote(int* device_any, int* device_all, int pshift) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  device_any[threadIdx.x >> pshift] = __any(tid - 77);
  device_all[threadIdx.x >> pshift] = __all(tid - 77);
}


TEST_CASE("Unit_AnyAll_CompileTest") {
  int warpSize, pshift;
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  warpSize = devProp.warpSize;

  int w = warpSize;
  pshift = 0;
  while (w >>= 1) ++pshift;

  INFO("WarpSize:: " << warpSize << " pShift: " << pshift);


  int anycount = 0;
  int allcount = 0;
  int Num_Threads_per_Block = 1024;
  int Num_Blocks_per_Grid = 1;
  int Num_Warps_per_Grid = (Num_Threads_per_Block * Num_Blocks_per_Grid) / warpSize;

  int* host_any = (int*)malloc(Num_Warps_per_Grid * sizeof(int));
  int* host_all = (int*)malloc(Num_Warps_per_Grid * sizeof(int));
  int* device_any;
  int* device_all;
  HIP_CHECK(hipMalloc((void**)&device_any, Num_Warps_per_Grid * sizeof(int)));
  HIP_CHECK(hipMalloc((void**)&device_all, Num_Warps_per_Grid * sizeof(int)));
  for (int i = 0; i < Num_Warps_per_Grid; i++) {
    host_any[i] = 0;
    host_all[i] = 0;
  }
  HIP_CHECK(hipMemcpy(device_any, host_any, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_all, host_all, sizeof(int), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(warpvote, dim3(Num_Blocks_per_Grid), dim3(Num_Threads_per_Block), 0, 0,
                     device_any, device_all, pshift);

  HIP_CHECK(
      hipMemcpy(host_any, device_any, Num_Warps_per_Grid * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(host_all, device_all, Num_Warps_per_Grid * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < Num_Warps_per_Grid; i++) {
    INFO("Warp Number: " << i << " __any: " << host_any[i] << " __all: " << host_all[i]);

    if (host_all[i] != 1) ++allcount;
    if (host_any[i] != 1) ++anycount;
  }

  HIP_CHECK(hipFree(device_any));
  HIP_CHECK(hipFree(device_all));
  REQUIRE(anycount == 0);
  REQUIRE(allcount == 1);
}
