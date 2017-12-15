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
 * BUILD: %t %s ../test_common.cpp 
 * RUN: %t
 * HIT_END
 */

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>
#include "test_common.h"


#define NUM  1024
#define SIZE NUM * 8

__global__ void Alloc(uint64_t *Ptr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Ptr[tid] = (uint64_t)malloc(128);
}

__global__ void Free(uint64_t *Ptr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    free((void*)Ptr[tid]);
}

int main()
{
    uint64_t *hPtr, *dPtr;
    hPtr = new uint64_t[NUM];
    for(uint32_t i=0;i<NUM;i++) {
        hPtr[i] = 1;
    }
    int devCnt = 0;
    hipGetDeviceCount(&devCnt);
    for(int i=0;i<devCnt;i++){
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipMalloc((void**)&dPtr, SIZE));
        HIPCHECK(hipMemcpy(dPtr, hPtr, SIZE, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(Alloc, dim3(1,1,1), dim3(NUM,1,1), 0, 0, dPtr);
        HIPCHECK(hipMemcpy(hPtr, dPtr, SIZE, hipMemcpyDeviceToHost));
        HIPASSERT(hPtr[0] != 0);
        hipLaunchKernelGGL(Free, dim3(1,1,1), dim3(NUM,1,1), 0, 0, dPtr);
        HIPCHECK(hipFree(dPtr));
        for(uint32_t i=1;i<NUM;i++) {
            HIPASSERT(hPtr[i] == hPtr[i-1] + 4096);
        }
    }
  passed();
}
