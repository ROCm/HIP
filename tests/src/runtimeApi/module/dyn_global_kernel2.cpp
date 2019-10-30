/*
Copyright (c) 2017-present Advanced Micro Devices, Inc. All rights reserved.

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

#if __HIP__
__hip_pinned_shadow__
#endif
texture<float, 2, hipReadModeElementType> texSingleVar;

__device__ float dynGlobal;

#if __HIP__
__hip_pinned_shadow__
#endif
texture<float, 2, hipReadModeElementType> texGlobal;

#if __HIP__
__hip_pinned_shadow__
#endif
texture<float, 2, hipReadModeElementType> statTexGlobal;

#if __HIP__
__hip_pinned_shadow__
#endif
texture<float, 2, hipReadModeElementType> statDynTexGlobal;

__hip_pinned_shadow__ int shadowGlobal;

extern "C" __global__ void tex2dKernel1(float* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texSingleVar, x, y);
}

extern "C" __global__ void tex2dKernel2(float* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texGlobal, x, y);
}

extern "C" __global__ void tex2dKernel3(float* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(statTexGlobal, x, y);
}

extern "C" __global__ void tex2dKernel4(float* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(statDynTexGlobal, x, y);
}
