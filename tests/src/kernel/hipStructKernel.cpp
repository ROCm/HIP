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

#include "test_common.h"
#include <hip/hip_runtime.h>
#include <stdio.h>

#define LEN 32
#define GROUP_SIZE 16


struct A
{
  int a[LEN];
};

__global__ void myKernel(struct A strucA, int *B) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    B[tid] = strucA.a[tid];
}

int main() {
    struct A A_h1;
    int *B_d1, *B_h1;

    //hipMalloc((void **)&A_h1, sizeof(struct A));
    for (int i=0; i<LEN; i++){
        A_h1.a[i] = i;
    }

    hipMalloc((void **)&B_d1, LEN*sizeof(int));
    hipHostMalloc((void **)&B_h1, LEN*sizeof(int));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(myKernel),
        dim3(LEN, 1, 1),
        dim3(GROUP_SIZE, 1, 1),
        0,
        0,
        A_h1,
        B_d1 );

    hipMemcpy(B_h1, B_d1, LEN*sizeof(int), hipMemcpyDeviceToHost);

    for (int i=0; i<LEN; i++) {
        //printf("A_h1.a[%d]: %d\tB_h1[%d]: %d\n", i, A_h1.a[i], i, B_h1[i]);
        HIPASSERT(A_h1.a[i] == B_h1[i]);
    }


    hipFree((void **)&B_d1);
    hipFree((void **)&B_h1);
    passed();
}

