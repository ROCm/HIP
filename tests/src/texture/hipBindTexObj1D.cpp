/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "test_common.h"
#include <iostream>
#include <string.h>

#define N 512
using namespace std;

texture<int, hipTextureType1D, hipReadModeElementType> tex;

bool testResult = true;

__global__ void kernel(int *out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  out[x] = tex1Dfetch(tex, x);
}

void runTest(void);

int main(int argc, char **argv) {
  runTest();

  if (testResult) {
    passed();
  } else {
    exit(EXIT_FAILURE);
  }
}

void runTest() {
  string out;
  int *tex_buf;
  int val[N], i, output[N];
  size_t size = 0;

  for (i = 0; i < N; i++) {
    val[i] = i;
    output[i] = 0;
  }
  hipChannelFormatDesc chan_desc =
      hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindUnsigned);

  hipMalloc(&tex_buf, N * sizeof(int));

  hipMemcpy(tex_buf, val, N * sizeof(int), hipMemcpyHostToDevice);

  tex.addressMode[0] = hipAddressModeWrap;
  tex.filterMode = hipFilterModeLinear;
  tex.normalized = true;

  hipBindTexture(&size, &tex, (void *)tex_buf, &chan_desc, N * sizeof(int));

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(N / dimBlock.x, 1, 1);

  hipLaunchKernelGGL(kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, output);

  hipDeviceSynchronize();

  hipMemcpy(output, tex_buf, N * sizeof(int), hipMemcpyDeviceToHost);

  for (i = 0; i < N; i++) {
    if (output[i] != val[i]) {
      testResult = false;
      return;
    }
  }
  hipUnbindTexture(&tex);
  hipFree(tex_buf);
}
