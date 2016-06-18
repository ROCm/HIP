/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

#include"hip_runtime.h"
#include"test_common.h"
#include"hip_runtime_api.h"
#include<iostream>

__global__ void vAdd(hipLaunchParm lp, float *a){}

int main()
{
    float *Ad;
    hipMalloc((void**)&Ad, 1024);
    hipLaunchKernel(vAdd, 1024, 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, 1024, dim3(1), 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad);
    passed();
}
