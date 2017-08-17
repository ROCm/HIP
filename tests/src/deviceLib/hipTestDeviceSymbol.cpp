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

/* HIT_START
 * BUILD: %t %s
 * RUN: %t
 * HIT_END
 */

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include"test_common.h"
#include<iostream>

#define NUM 1024
#define SIZE 1024*4

__device__ int globalIn[NUM];
__device__ int globalOut[NUM];

__global__ void Assign(hipLaunchParm lp, int* Out)
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Out[tid] = globalIn[tid];
    globalOut[tid] = globalIn[tid];
}

int main()
{
    int *A, *Am, *B, *Ad, *C, *Cm;
    A = new int[NUM];
    B = new int[NUM];
    C = new int[NUM];
    for(int i=0;i<NUM;i++) {
        A[i] = -1*i;
        B[i] = 0;
        C[i] = 0;
    }

    hipMalloc((void**)&Ad, SIZE);
    hipHostMalloc((void**)&Am, SIZE);
    hipHostMalloc((void**)&Cm, SIZE);
    for(int i=0;i<NUM;i++) {
        Am[i] = -1*i;
        Cm[i] = 0;
    }

    hipStream_t stream;
    hipStreamCreate(&stream);
    hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), Am, SIZE, 0, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);
    hipLaunchKernel(Assign, dim3(1,1,1), dim3(NUM,1,1), 0, 0, Ad);
    hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost);
    hipMemcpyFromSymbolAsync(Cm, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    for(int i=0;i<NUM;i++) {
        assert(Am[i] == B[i]);
        assert(Am[i] == Cm[i]);
    }

    for(int i=0;i<NUM;i++) {
        A[i] = -2*i;
        B[i] = 0;
    }

    hipMemcpyToSymbol(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice);
    hipLaunchKernel(Assign, dim3(1,1,1), dim3(NUM,1,1), 0, 0, Ad);
    hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost);
    hipMemcpyFromSymbol(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost);
    for(int i=0;i<NUM;i++) {
        assert(A[i] == B[i]);
        assert(A[i] == C[i]);
    }

    for(int i=0;i<NUM;i++) {
        A[i] = -3*i;
        B[i] = 0;
    }

    hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);
    hipLaunchKernel(Assign, dim3(1,1,1), dim3(NUM,1,1), 0, 0, Ad);
    hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost);
    hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    for(int i=0;i<NUM;i++) {
        assert(A[i] == B[i]);
        assert(A[i] == C[i]);
    }
    hipHostFree(Am);
    hipHostFree(Cm);
    hipFree(Ad);
    delete[] A;
    delete[] B;
    delete[] C;
    passed();
}
