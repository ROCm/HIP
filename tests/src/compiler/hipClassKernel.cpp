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

#include "compiler/hipClassKernel.h"

 // check sizeof empty class in kernel
__global__ void 
 emptyClassKernel(bool* result_ecd){
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   result_ecd[tid] = (sizeof(testClassEmpty) == 1);
}


 int main() {
    bool *result_ecd,*result_ech;
    size_t NBOOL = BLOCKS * sizeof(bool);
    
    HIPCHECK(hipMalloc((void**)&result_ecd,
                        NBOOL));
    HIPCHECK(hipMemset(result_ecd, 
                       false, 
                       NBOOL));

    HIPCHECK(hipHostMalloc((void**)&result_ech,
                           NBOOL,
                           hipHostMallocDefault));
    
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(emptyClassKernel),
        dim3(BLOCKS),
        dim3(THREADS_PER_BLOCK),
        0,
        0,
        result_ecd);

    HIPCHECK(hipMemcpy(result_ech, 
                       result_ecd,
                       BLOCKS*sizeof(bool), 
                       hipMemcpyDeviceToHost));

    // validation on host side
    for(int i = 0; i < BLOCKS; i++){
        HIPASSERT(result_ech[i] == true);
    }
    
    hipFree((void **)&result_ech);
    hipFree((void **)&result_ecd);
    
    passed();
}