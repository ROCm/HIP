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
#include <hip/device_functions.h>

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>


#define WIDTH 8
#define HEIGHT 8

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1

template <typename T> unsigned lastbit(T a) {
  if (a == 0) return 0;
  unsigned pos = 1;
  while ((a & 1) != 1) {
    a >>= 1;
    pos++;
  }
  return pos;
}

__global__ void ffs_HIP_kernel(unsigned int* a, unsigned int* b, unsigned int* c,
                               unsigned long long int* d, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = __ffs(b[i]);
    c[i] = __ffsll(d[i]);
  }
}

TEST_CASE("Unit_ffs") {
  using namespace std;

  unsigned int* hostA;
  unsigned int* hostB;
  unsigned int* hostC;
  unsigned long long int* hostD;

  unsigned int* deviceA;
  unsigned int* deviceB;
  unsigned int* deviceC;
  unsigned long long int* deviceD;

  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  INFO("System minor : " << devProp.minor);
  INFO("System major : " << devProp.major);
  INFO("agent prop name : " << devProp.name);

  INFO("hip Device prop succeeded");


  int i;
  int errors;

  hostA = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostB = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostC = (unsigned int*)malloc(NUM * sizeof(unsigned int));
  hostD = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = i;
    hostD[i] = 1099511627776 + i;
  }

  HIP_CHECK(hipMalloc((void**)&deviceA, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceB, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceC, NUM * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&deviceD, NUM * sizeof(unsigned long long int)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(deviceD, hostD, NUM * sizeof(unsigned long long int), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(ffs_HIP_kernel,
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB,
                     deviceC, deviceD, WIDTH, HEIGHT);


  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hostC, deviceC, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != lastbit(hostB[i])) {
      INFO("Mismatch: " << hostA[i] << " - " << lastbit(hostB[i]));
      errors++;
    }
  }

  for (i = 0; i < NUM; i++) {
    if (hostC[i] != lastbit(hostD[i])) {
      INFO("Mismatch: " << hostC[i] << " - " << lastbit(hostD[i]));
      errors++;
    }
  }

  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));
  HIP_CHECK(hipFree(deviceC));
  HIP_CHECK(hipFree(deviceD));


  free(hostA);
  free(hostB);
  free(hostC);
  free(hostD);

  REQUIRE(errors == 0);
}
