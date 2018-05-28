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

/*HIT_START
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

bool testResult = true;

__global__ void tex1d_kernel(float *val, hipTextureObject_t obj) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  val[k] = tex1Dfetch<float>(obj, k);
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

  // Allocating the required buffer on gpu device
  float *tex_buf, *tex_buf_check;
  float val[N], output[N];
  int i;
  for (i = 0; i < N; i++)
    val[i] = (i + 1) * (i + 1);
  hipMalloc(&tex_buf, N * sizeof(float));

  hipMalloc(&tex_buf_check, N * sizeof(float));

  hipMemcpy(tex_buf, val, N * sizeof(float), hipMemcpyHostToDevice);

  hipMemset(tex_buf_check, 0, N * sizeof(float));
  hipResourceDesc res_lin;

  memset(&res_lin, 0, sizeof(res_lin));

  res_lin.resType = hipResourceTypeLinear;
  res_lin.res.linear.devPtr = tex_buf;
  res_lin.res.linear.desc.f = hipChannelFormatKindFloat;
  res_lin.res.linear.desc.x = 32;
  res_lin.res.linear.sizeInBytes = N * sizeof(float);

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = hipReadModeElementType;

  // Creating texture object

  hipTextureObject_t tex_obj = 0;

  hipCreateTextureObject(&tex_obj, &res_lin, &tex_desc, NULL);

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(N / dimBlock.x, 1, 1);

  for (i = 0; i < N; i++)
    output[i] = 0;

  hipLaunchKernelGGL(tex1d_kernel, dim3(dimGrid), dim3(dimBlock), 0, 0,
                     tex_buf_check, tex_obj);
  hipDeviceSynchronize();

  hipMemcpy(output, tex_buf_check, N * sizeof(float), hipMemcpyDeviceToHost);

  for (i = 0; i < N; i++)
    if (output[i] != val[i]) {
      testResult = false;
    }

  hipDestroyTextureObject(tex_obj);
  hipFree(tex_buf);
  hipFree(tex_buf_check);
}
