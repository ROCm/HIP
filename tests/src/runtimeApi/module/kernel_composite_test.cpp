/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip/hip_runtime.h"
#define GLOBAL_BUF_SIZE 2048

__device__ float deviceGlobalFloat;
__device__ int   deviceGlobalInt1;
__device__ int   deviceGlobalInt2;
__device__ short deviceGlobalShort;
__device__ char  deviceGlobalChar;

__device__ int getSquareOfGlobalFloat() {
  return static_cast<int>(deviceGlobalFloat*deviceGlobalFloat);
}

extern "C" __global__ void testWeightedCopy(int* a, int* b) {
  int tx = hipThreadIdx_x;
  b[tx] = deviceGlobalInt1*a[tx] + deviceGlobalInt2 +
  static_cast<int>(deviceGlobalShort) + static_cast<int>(deviceGlobalChar)
  + getSquareOfGlobalFloat();
}
