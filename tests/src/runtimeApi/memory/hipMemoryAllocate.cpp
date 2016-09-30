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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include"test_common.h"

#define SIZE 1024*1024*256

int main(){
    float *Ad, *B, *Bd, *Bm, *C, *Cd, *ptr_0;
    B = (float*)malloc(SIZE);
    hipMalloc((void**)&Ad, SIZE);
    hipHostMalloc((void**)&B, SIZE);
    hipHostMalloc((void**)&Bd, SIZE, hipHostMallocDefault);
    hipHostMalloc((void**)&Bm, SIZE, hipHostMallocMapped);
    hipHostMalloc((void**)&C, SIZE, hipHostMallocMapped);

    hipHostGetDevicePointer((void**)&Cd, C, 0/*flags*/);

    HIPCHECK_API(hipMalloc((void**)&ptr_0,0), hipSuccess);

    HIPCHECK_API(hipFree(Ad) ,  hipSuccess);
    HIPCHECK_API(hipHostFree(Ad) , hipErrorInvalidValue);

    HIPCHECK_API(hipFree(B)  , hipErrorInvalidDevicePointer); // try to hipFree on malloced memory
    HIPCHECK_API(hipFree(Bd) , hipErrorInvalidDevicePointer);
    HIPCHECK_API(hipFree(Bm) , hipErrorInvalidDevicePointer);
    HIPCHECK_API(hipFree(ptr_0) , hipSuccess);
    HIPCHECK_API(hipHostFree(Bd) , hipSuccess);
    HIPCHECK_API(hipHostFree(Bm) , hipSuccess);

    HIPCHECK_API(hipFree(C) , hipErrorInvalidDevicePointer);
    HIPCHECK_API(hipHostFree(C) , hipSuccess);


    HIPCHECK_API(hipFree(NULL) , hipSuccess);
    HIPCHECK_API(hipHostFree(NULL) , hipSuccess);
    passed();
}
