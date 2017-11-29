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


#include "hip/hip_runtime.h"
#include "test_common.h"
#include <stdio.h>

#define ITER 1<<20
#define SIZE 1024*1024*sizeof(int)

__global__ void Iter(hipLaunchParm lp, int *Ad){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(tx == 0){
        for(int i=0;i<ITER;i++){
            Ad[tx] += 1;
        }
    }
}

int main(){
    int A=0, *Ad;
    hipMalloc((void**)&Ad, SIZE);
    hipMemcpy(Ad, &A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernel(HIP_KERNEL_NAME(Iter), dim3(1), dim3(1), 0, 0, Ad);
    hipMemcpy(&A, Ad, SIZE, hipMemcpyDeviceToHost);
    passed();
}
