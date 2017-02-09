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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM all
 * RUN: %t
 * HIT_END
 */

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>

#define HIP_ASSERT(status) assert(hipSuccess == status);

#define NUM  1024
#define SIZE NUM * 8

__global__ void Alloc(hipLaunchParm lp, uint64_t *Ptr) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Ptr[tid] = (uint64_t)malloc(128);
}

__global__ void Free(hipLaunchParm lp, uint64_t *Ptr) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    free((void*)Ptr[tid]);
}

int main()
{
    uint64_t *hPtr, *dPtr;
    hPtr = new uint64_t[NUM];
    for(uint32_t i=0;i<NUM;i++) {
        hPtr[i] = 1;
    }
    int devCnt;
    hipGetDeviceCount(&devCnt);
    for(uint32_t i=0;i<devCnt;i++){
        HIP_ASSERT(hipSetDevice(i));
        HIP_ASSERT(hipMalloc((void**)&dPtr, SIZE));
        HIP_ASSERT(hipMemcpy(dPtr, hPtr, SIZE, hipMemcpyHostToDevice));
        hipLaunchKernel(Alloc, dim3(1,1,1), dim3(NUM,1,1), 0, 0, dPtr);
        HIP_ASSERT(hipMemcpy(hPtr, dPtr, SIZE, hipMemcpyDeviceToHost));
        assert(hPtr[0] != 0);
        hipLaunchKernel(Free, dim3(1,1,1), dim3(NUM,1,1), 0, 0, dPtr);
        HIP_ASSERT(hipFree(dPtr));
        for(uint32_t i=1;i<NUM;i++) {
            assert(hPtr[i] == hPtr[i-1] + 4096);
        }
    }
}
