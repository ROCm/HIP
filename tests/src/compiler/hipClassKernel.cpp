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

// check sizeof empty class is 1
__global__ void
emptyClassKernel(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  result_ecd[tid] = (sizeof(testClassEmpty) == 1);
}

// check object addresses are not same
__global__ void
emptyClassKernelObj(bool* result_ecd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  testClassEmpty ob1, ob2;
  result_ecd[tid] = (&ob1 != &ob2);
}


int main() {
  bool *result_ecd, *result_eob, *result_ech, *result_eoh;
  size_t NBOOL = BLOCKS * sizeof(bool);

  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&result_ech),
                         NBOOL,
                         hipHostMallocDefault));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&result_ecd),
                     NBOOL));
  HIPCHECK(hipMemset(result_ecd,
                       false,
                       NBOOL));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(emptyClassKernel),
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
  for (int i = 0; i < BLOCKS; i++) {
    HIPASSERT(result_ech[i] == true);
    }


  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&result_eoh),
                         NBOOL,
                         hipHostMallocDefault));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&result_eob),
                     NBOOL));
  HIPCHECK(hipMemset(result_eob,
                     false,
                     NBOOL));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(emptyClassKernelObj),
                     dim3(BLOCKS),
                     dim3(THREADS_PER_BLOCK),
                     0,
                     0,
                     result_eob);
  HIPCHECK(hipMemcpy(result_eoh,
                       result_eob,
                       BLOCKS*sizeof(bool),
                       hipMemcpyDeviceToHost));

    // validation on host side
  for (int i = 0; i < BLOCKS; i++) {
    HIPASSERT(result_eoh[i] == true);
    }


  HIPCHECK(hipHostFree(result_ech));
  HIPCHECK(hipFree(result_ecd));
  HIPCHECK(hipFree(result_eob));
  HIPCHECK(hipHostFree(result_eoh));

  passed();
}
