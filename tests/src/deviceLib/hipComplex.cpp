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


#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip/hcc_detail/hip_complex.h"

#define LEN 64
#define SIZE 64<<2

__global__  void getSqAbs(hipLaunchParm lp, float *A, float *B, float *C){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    C[tx] = hipCsqabsf(make_hipFloatComplex(A[tx], B[tx]));
}

int main(){
    float *A, *Ad, *B, *Bd, *C, *Cd;
    A = new float[LEN];
    B = new float[LEN];
    C = new float[LEN];
    for(uint32_t i=0;i<LEN;i++){
        A[i] = i*1.0f;
        B[i] = i*1.0f;
        C[i] = i*1.0f;
    }

    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernel(getSqAbs, dim3(1), dim3(LEN), 0, 0, Ad, Bd, Cd);
    hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost);
    std::cout<<A[11]<<" "<<B[11]<<" "<<C[11]<<std::endl;
}
