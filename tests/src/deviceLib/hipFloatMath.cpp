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

/* HIT_START
 * BUILD: %t %s
 * RUN: %t
 * HIT_END
 */

#include "test_common.h"
#include <hip/device_functions.h>

#define LEN 512
#define SIZE LEN<<2



__global__ void floatMath(hipLaunchParm lp, float *In, float *Out) {
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  Out[tid] = __cosf(In[tid]);
  Out[tid] = __exp10f(Out[tid]);
  Out[tid] = __expf(Out[tid]);
  Out[tid] = __frsqrt_rn(Out[tid]);
  Out[tid] = __fsqrt_rd(Out[tid]);
  Out[tid] = __fsqrt_rn(Out[tid]);
  Out[tid] = __fsqrt_ru(Out[tid]);
  Out[tid] = __fsqrt_rz(Out[tid]);
  Out[tid] = __log10f(Out[tid]);
  Out[tid] = __log2f(Out[tid]);
  Out[tid] = __logf(Out[tid]);
  Out[tid] = __powf(2.0f, Out[tid]);
  __sincosf(Out[tid], &In[tid], &Out[tid]);
  Out[tid] = __sinf(Out[tid]);
  Out[tid] = __cosf(Out[tid]);
  Out[tid] = __tanf(Out[tid]);
}

int main(){
  float *Inh, *Outh, *Ind, *Outd;
  hipMalloc((void**)&Ind, SIZE);
  hipMalloc((void**)&Outd, SIZE);
  hipLaunchKernel(floatMath, dim3(LEN,1,1), dim3(1,1,1), 0, 0, Ind, Outd);
  passed();
}
