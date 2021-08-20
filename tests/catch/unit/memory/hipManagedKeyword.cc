/*
   Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

/*
   This testcase verifies the hipManagedKeyword basic scenario
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define N 1048576
__managed__ float A[N];   // Accessible by ALL CPU and GPU functions !!!
__managed__ float B[N];
__managed__  int  x = 0;

__global__ void add(const float *A, float *B) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride)
    B[i] = A[i] + B[i];
}

__global__ void GPU_func() {
  x++;
}

TEST_CASE("Unit_hipManagedKeyword_SingleGpu") {
  for (int i = 0; i < N; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(blockSize, 1, 1);
  hipLaunchKernelGGL(add, dimGrid, dimBlock, 0, 0, static_cast<const float*>(A),
                     static_cast<float*>(B));

  hipDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(B[i]-3.0f));

  REQUIRE(maxError == 0.0f);
}

TEST_CASE("Unit_hipManagedKeyword_MultiGpu") {
  int numDevices = 0;
  hipGetDeviceCount(&numDevices);

  for (int i = 0; i < numDevices; i++) {
    hipSetDevice(i);
    GPU_func<<< 1, 1 >>>();
    hipDeviceSynchronize();
  }
  REQUIRE(x == numDevices);
}
