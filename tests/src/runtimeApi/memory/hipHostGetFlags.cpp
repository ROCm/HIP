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
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include"test_common.h"
#include<malloc.h>

#define LEN 1024*1024
#define SIZE LEN*sizeof(float)

__global__ void Add(hipLaunchParm lp, float *Ad, float *Bd, float *Cd){
int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
Cd[tx] = Ad[tx] + Bd[tx];
}

int main(){
float *A, *B, *C, *D;
float *Ad, *Bd, *Cd, *Dd;
unsigned int FlagA, FlagB, FlagC;
FlagA = hipHostMallocWriteCombined | hipHostMallocMapped;
FlagB = hipHostMallocWriteCombined | hipHostMallocMapped;
FlagC = hipHostMallocMapped;
hipDeviceProp_t prop;
int device;
HIPCHECK(hipGetDevice(&device));
HIPCHECK(hipGetDeviceProperties(&prop, device));
if(prop.canMapHostMemory != 1){
std::cout<<"Exiting..."<<std::endl;
}
HIPCHECK(hipHostMalloc((void**)&A, SIZE, hipHostMallocWriteCombined | hipHostMallocMapped));
HIPCHECK(hipHostMalloc((void**)&B, SIZE, hipHostMallocWriteCombined | hipHostMallocMapped));
HIPCHECK(hipHostMalloc((void**)&C, SIZE, hipHostMallocMapped));

HIPCHECK(hipHostMalloc((void**)&D, SIZE, hipHostMallocDefault));

unsigned int flagA, flagB, flagC;
HIPCHECK(hipHostGetDevicePointer((void**)&Ad, A, 0));
HIPCHECK(hipHostGetDevicePointer((void**)&Bd, B, 0));
HIPCHECK(hipHostGetDevicePointer((void**)&Cd, C, 0));
HIPCHECK(hipHostGetDevicePointer((void**)&Dd, D, 0));
HIPCHECK(hipHostGetFlags(&flagA, A));
HIPCHECK(hipHostGetFlags(&flagB, B));
HIPCHECK(hipHostGetFlags(&flagC, C));

for(int i=0;i<LEN;i++){
A[i] = 1.0f;
B[i] = 2.0f;
}

dim3 dimGrid(LEN/512,1,1);
dim3 dimBlock(512,1,1);

hipLaunchKernel(HIP_KERNEL_NAME(Add), dimGrid, dimBlock, 0, 0, Ad, Bd, Cd);

HIPCHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));  // Note this really HostToHost not DeviceToHost, since memory is mapped...
HIPASSERT(C[10] == 3.0f);
HIPASSERT(flagA == FlagA);
HIPASSERT(flagB == FlagB);
HIPASSERT(flagC == FlagC);
passed();

}
