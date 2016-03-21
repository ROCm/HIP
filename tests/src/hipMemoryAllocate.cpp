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
#include"test_common.h"

#define SIZE 1024*1024*256

int main(){
    float *Ad, *B, *Bd, *Bm, *C, *Cd;
    B = (float*)malloc(SIZE);
    hipMalloc((void**)&Ad, SIZE);
    hipHostMalloc((void**)&B, SIZE);
    hipHostMalloc((void**)&Bd, SIZE, hipHostMallocDefault);
    hipHostMalloc((void**)&Bm, SIZE, hipHostMallocMapped);
    hipHostMalloc((void**)&C, SIZE, hipHostMallocMapped);
    hipHostGetDevicePointer((void**)&Cd, C, SIZE);

    HIPASSERT(hipFree(Ad) == hipSuccess);
    HIPASSERT(hipHostFree(Ad) == hipErrorInvalidDevicePointer);

    HIPASSERT(hipFree(B) == hipErrorInvalidDevicePointer);
    HIPASSERT(hipFree(Bd) == hipErrorInvalidDevicePointer);
    HIPASSERT(hipFree(Bm) == hipErrorInvalidDevicePointer);
    HIPASSERT(hipHostFree(Bd) == hipSuccess);
    HIPASSERT(hipHostFree(Bm) == hipSuccess);

    HIPASSERT(hipFree(C) == hipErrorInvalidDevicePointer);
    HIPASSERT(hipFree(Cd) == hipErrorInvalidDevicePointer);
    HIPASSERT(hipHostFree(C) == hipSuccess);
    HIPASSERT(hipHostFree(Cd) == hipErrorInvalidDevicePointer);
    HIPASSERT(hipFree(Cd) == hipErrorInvalidDevicePointer);

    HIPASSERT(hipFree(NULL) == hipErrorInvalidDevicePointer);
    HIPASSERT(hipHostFree(NULL) == hipErrorInvalidDevicePointer);
    passed();
}
