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

#include <stdio.h>
#include <hip/hip_fp16.h>
#include "hip_runtime_api.h"

#define DSIZE 4
#define SCF 0.5f
#define nTPB 256
__global__ void half_scale_kernel(hipLaunchParm lp, float *din, float *dout, int dsize){

  int idx = hipThreadIdx_x+ hipBlockDim_x*hipBlockIdx_x;
  if (idx < dsize){
    __half scf = __float2half(SCF);
    __half kin = __float2half(din[idx]);
    __half kout;

    kout = __hmul(kin, scf);

//    kout = cvt_float_to_half(cvt_half_to_float(kin)*cvt_half_to_float(scf));

    dout[idx] = __half2float(kout);
    }
}

int main(){

  float *hin, *hout, *din, *dout;
  hin  = (float *)malloc(DSIZE*sizeof(float));
  hout = (float *)malloc(DSIZE*sizeof(float));
  for (int i = 0; i < DSIZE; i++) hin[i] = i;
  hipMalloc(&din,  DSIZE*sizeof(float));
  hipMalloc(&dout, DSIZE*sizeof(float));
  hipMemcpy(din, hin, DSIZE*sizeof(float), hipMemcpyHostToDevice);
  hipLaunchKernel(half_scale_kernel, dim3((DSIZE+nTPB-1)/nTPB),dim3(nTPB), 0, 0, din, dout, DSIZE);
  hipMemcpy(hout, dout, DSIZE*sizeof(float), hipMemcpyDeviceToHost);
  for (int i = 0; i < DSIZE; i++) printf("%f\n", hout[i]);
  return 0;
}
