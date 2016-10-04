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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * Test for checking the functionality of
 * hipError_t hipDeviceSynchronize();
*/

#include "hip/hip_runtime.h"
#include<iostream>

#define _SIZE sizeof(int)*1024*1024
#define NUM_STREAMS 20
#define ITER 1<<10

__global__ void Iter(hipLaunchParm lp, int *Ad, int num){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(tx == 0){
        for(int i = 0; i<num;i++){
            Ad[tx] += 1;
        }
    }
}

int main(){
    int *A[NUM_STREAMS];
    int *Ad[NUM_STREAMS];
    hipStream_t stream[NUM_STREAMS];
    for(int i=0;i<NUM_STREAMS;i++){
        hipHostMalloc((void**)&A[i], _SIZE, hipHostMallocDefault);
        A[i][0] = 1;
        hipMalloc((void**)&Ad[i], _SIZE);
    }
    for(int i=0;i<NUM_STREAMS;i++){
        for(int j=0;j<ITER;j++){
        std::cout<<"Iter: "<<j<<std::endl;
        hipMemcpy(Ad[i], A[i], _SIZE, hipMemcpyHostToDevice);
        hipLaunchKernel(HIP_KERNEL_NAME(Iter), dim3(1), dim3(1), 0, 0, Ad[i], 1<<30);
        hipMemcpyAsync(A[i], Ad[i], _SIZE, hipMemcpyDeviceToHost);
        }
    }

    std::cout<<"Waitin..."<<std::endl;
    
    hipDeviceSynchronize();
}

