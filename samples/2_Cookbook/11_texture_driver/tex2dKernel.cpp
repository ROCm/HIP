/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

texture<char, hipTextureType2D, hipReadModeElementType> texChar;
texture<short, hipTextureType2D, hipReadModeElementType> texShort;
texture<int, hipTextureType2D, hipReadModeElementType> texInt;
texture<float, hipTextureType2D, hipReadModeElementType> texFloat;

texture<char4, hipTextureType2D, hipReadModeElementType> texChar4;
texture<short4, hipTextureType2D, hipReadModeElementType> texShort4;
texture<int4, hipTextureType2D, hipReadModeElementType> texInt4;
texture<float4, hipTextureType2D, hipReadModeElementType> texFloat4;

extern "C" __global__ void tex2dKernelChar(char* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texChar, x, y);
}

extern "C" __global__ void tex2dKernelShort(short* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texShort, x, y);
}

extern "C" __global__ void tex2dKernelInt(int* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texInt, x, y);
}

extern "C" __global__ void tex2dKernelFloat(float* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texFloat, x, y);
}

extern "C" __global__ void tex2dKernelChar4(char4* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texChar4, x, y);
}

extern "C" __global__ void tex2dKernelShort4(short4* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texShort4, x, y);
}

extern "C" __global__ void tex2dKernelInt4(int4* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texInt4, x, y);
}

extern "C" __global__ void tex2dKernelFloat4(float4* outputData, int width, int height) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    outputData[y * width + x] = tex2D(texFloat4, x, y);
}
