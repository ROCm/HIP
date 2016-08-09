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

#include"test_common.h"

#define LEN 1024*1024
#define SIZE LEN*sizeof(float)

__global__ void Add(hipLaunchParm lp, float *Ad, float *Bd, float *Cd){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Cd[tx] = Ad[tx] + Bd[tx];
}

int main(){
    float *A, *B, *C;
    float *Ad, *Bd, *Cd;

    hipDeviceProp_t prop;
    int device;
    HIPCHECK(hipGetDevice(&device));
    HIPCHECK(hipGetDeviceProperties(&prop, device));
    if(prop.canMapHostMemory != 1){
        std::cout<<"Exiting..."<<std::endl;
        failed("Does support HostPinned Memory");
    }

    HIPCHECK(hipHostMalloc((void**)&A, SIZE, hipHostMallocWriteCombined | hipHostMallocMapped));
    HIPCHECK(hipHostMalloc((void**)&B, SIZE, hipHostMallocDefault));
    HIPCHECK(hipHostMalloc((void**)&C, SIZE, hipHostMallocMapped));

    HIPCHECK(hipHostGetDevicePointer((void**)&Ad, A, 0));
    HIPCHECK(hipHostGetDevicePointer((void**)&Cd, C, 0));

    for(int i=0;i<LEN;i++){
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    HIPCHECK(hipMalloc((void**)&Bd, SIZE));
    HIPCHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

    dim3 dimGrid(LEN/512,1,1);
    dim3 dimBlock(512,1,1);

    hipLaunchKernel(HIP_KERNEL_NAME(Add), dimGrid, dimBlock, 0, 0, Ad, Bd, Cd);

    passed();

}
