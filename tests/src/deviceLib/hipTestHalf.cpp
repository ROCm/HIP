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
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip/hip_fp16.h"

#define hInf     0x7C00
#define hInfPK   0x7C007C00
#define h65504   0xF7FF
#define h65504PK 0xF7FFF7FF
#define h27      0x4EC0
#define h27PK    0x4EC04EC0
#define h7       0x4700
#define h7PK     0x47004700
#define h3       0x4200
#define h3PK     0x42004200
#define h1       0x3C00
#define h1PK     0x3C003C00
#define hPoint5     0x3800
#define hPoint5PK   0x38003800
#define hZero    0x0000
#define hNeg1    0xBC00
#define hNeg1PK 0xBC00BC00

struct holder{
union{
  __half a;
  unsigned short b;
};
};

__global__ void CheckHalf(hipLaunchParm lp, __half* In1, __half* In2, __half* In3, __half* Out){
  Out[0] = __hadd(In1[0], In2[0]);
  Out[1] = __hadd_sat(In1[1], In2[1]);
  Out[2] = __hfma(In1[2], In2[2],In3[2]);
  Out[3] = __hfma_sat(In1[3], In2[3], In3[3]);
  Out[4] = __hmul(In1[4], In2[4]);
  Out[5] = __hmul_sat(In1[5], In2[5]);
  Out[6] = __hneg(In1[6]);
  Out[7] = __hsub(In1[7], In2[7]);
  Out[8] = __hsub_sat(In1[8], In2[8]);
  Out[9] = hdiv(In1[9], In2[9]);
}


int main(){

}
