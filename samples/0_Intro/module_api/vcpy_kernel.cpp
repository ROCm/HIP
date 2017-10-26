/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#define ARRAY_SIZE (16)

__device__ float myDeviceGlobal;
//extern "C" __device__ int * myDeviceGlobalPtr; // Unreferences globals are eliminated.
__constant__ float myDeviceGlobalArray[16];;


extern "C" __global__ void hello_world(hipLaunchParm lp, const float *a, float *b)
{
    int tx = hipThreadIdx_x;
    b[tx] = a[tx];
}

extern "C" __global__ void test_device_globals(hipLaunchParm lp, const float *a, float *b)
{
    int tx = hipThreadIdx_x;
    b[tx] = a[tx] + myDeviceGlobal+ myDeviceGlobalArray[tx%ARRAY_SIZE] ;
}

#define TEST_PLATFORM_GLOBAL 0
#if TEST_PLATFORM_GLOBAL
float myPlatformGlobal;

extern "C" __global__ void test_platform_globals(hipLaunchParm lp, float *a, float *b)
{
    int tx = hipThreadIdx_x;
    b[tx] = a[tx] + myPlatformGlobal;
}
#endif
