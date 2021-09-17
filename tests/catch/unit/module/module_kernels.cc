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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_kernels.hh>
#include "hip/hip_runtime.h"

#define GLOBAL_BUF_SIZE 2048
#define ARRAY_SIZE (16)

texture<float, 2, hipReadModeElementType> ftex;
texture<int, 2, hipReadModeElementType> itex;
texture<uint16_t, 2, hipReadModeElementType> stex;
texture<char, 2, hipReadModeElementType> ctex;

__device__ int deviceGlobal = 1;
__managed__ int x = 10;
__device__ float myDeviceGlobal;
__device__ float myDeviceGlobalArray[16];


__device__ float deviceGlobalFloat;
__device__ int   deviceGlobalInt1;
__device__ int   deviceGlobalInt2;
__device__ uint16_t deviceGlobalShort;
__device__ char  deviceGlobalChar;

extern "C" __global__ void tex2dKernelFloat(float* outputData,
                                       int width, int height) {
  int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  if ((x < width) && (y < width)) {
    outputData[y * width + x] = tex2D(ftex, x, y);
  }
}

extern "C" __global__ void tex2dKernelInt(int* outputData,
                                          int width, int height) {
  int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  if ((x < width) && (y < width)) {
    outputData[y * width + x] = tex2D(itex, x, y);
  }
}

extern "C" __global__ void tex2dKernelInt16(uint16_t* outputData,
                                          int width, int height) {
  int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  if ((x < width) && (y < width)) {
    outputData[y * width + x] = tex2D(stex, x, y);
  }
}

extern "C" __global__ void tex2dKernelInt8(char* outputData,
                                          int width, int height) {
  int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  if ((x < width) && (y < width)) {
    outputData[y * width + x] = tex2D(ctex, x, y);
  }
}

extern "C" __global__ void matmulK(int clockrate, int* A, int* B, int* C,
                                   int N) {
  int ROW = blockIdx.y*blockDim.y+threadIdx.y;
  int COL = blockIdx.x*blockDim.x+threadIdx.x;
  int tmpSum = 0;
  if ((ROW < N) && (COL < N)) {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
      tmpSum += A[ROW * N + i] * B[i * N + COL];
    }
    C[ROW * N + COL] = tmpSum;
  }
}

extern "C" __global__ void KernelandExtraParams(int* A, int* B, int* C,
  int *D, int N) {
  int ROW = blockIdx.y*blockDim.y+threadIdx.y;
  int COL = blockIdx.x*blockDim.x+threadIdx.x;
  int tmpSum = 0;
  if (ROW < N && COL < N) {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
      tmpSum += A[ROW * N + i] * B[i * N + COL];
    }
  }
  C[ROW * N + COL] = tmpSum;
  D[ROW * N + COL] = tmpSum;
}

extern "C" __global__ void SixteenSecKernel(int clockrate) {
  HipTest::waitKernel(16, clockrate);
}

extern "C" __global__ void TwoSecKernel(int clockrate) {
  if (deviceGlobal == 0x2222) {
    deviceGlobal = 0x3333;
  }

  HipTest::waitKernel(2, clockrate);

  if (deviceGlobal != 0x3333) {
    deviceGlobal = 0x5555;
  }
}

extern "C" __global__ void FourSecKernel(int clockrate) {
  if (deviceGlobal == 1) {
    deviceGlobal = 0x2222;
  }

  HipTest::waitKernel(4, clockrate);

  if (deviceGlobal == 0x2222) {
    deviceGlobal = 0x4444;
  }
}

extern "C" __global__ void GPU_func() {
  x++;
}


__device__ int getSquareOfGlobalFloat() {
  return static_cast<int>(deviceGlobalFloat*deviceGlobalFloat);
}

extern "C" __global__ void testWeightedCopy(int* a, int* b) {
  int tx = hipThreadIdx_x;
  b[tx] = deviceGlobalInt1*a[tx] + deviceGlobalInt2 +
  static_cast<int>(deviceGlobalShort) + static_cast<int>(deviceGlobalChar)
  + getSquareOfGlobalFloat();
}


extern "C" __global__ void hello_world(const float* a, float* b) {
    int tx = hipThreadIdx_x;
    b[tx] = a[tx];
}

extern "C" __global__ void test_globals(const float* a, float* b) {
    int tx = hipThreadIdx_x;
    b[tx] = a[tx] + myDeviceGlobal + myDeviceGlobalArray[tx % ARRAY_SIZE];
}

extern "C" __global__ void EmptyKernel() {
}
