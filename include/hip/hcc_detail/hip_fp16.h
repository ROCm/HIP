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

#ifndef HIP_FP16_H
#define HIP_FP16_H

#include "hip/hip_runtime.h"

#if __clang_major__ == 4

typedef __fp16 __half;

typedef struct __attribute__((aligned(4))){
  union {
    __half p[2];
    unsigned int q;
  };
} __half2;

extern "C" __half __hip_hc_ir_hadd_half(__half, __half);
extern "C" __half __hip_hc_ir_hfma_half(__half, __half, __half);
extern "C" __half __hip_hc_ir_hmul_half(__half, __half);
extern "C" __half __hip_hc_ir_hsub_half(__half, __half);

__device__ static inline __half __hadd(const __half a, const __half b) {
  return __hip_hc_ir_hadd_half(a, b);
}

__device__ static inline __half __hadd_sat(__half a, __half b) {
  return __hip_hc_ir_hadd_half(a, b);
}

__device__ static inline __half __hfma(__half a, __half b, __half c) {
  return __hip_hc_ir_hfma_half(a, b, c);
}

__device__ static inline __half __hfma_sat(__half a, __half b, __half c) {
  return __hip_hc_ir_hfma_half(a, b, c);
}

__device__ static inline __half __hmul(__half a, __half b) {
  return __hip_hc_ir_hmul_half(a, b);
}

__device__ static inline __half __hmul_sat(__half a, __half b) {
  return __hip_hc_ir_hmul_half(a, b);
}

__device__ static inline __half __hneg(__half a) {
  return -a;
}

__device__ static inline __half __hsub(__half a, __half b) {
  return __hip_hc_ir_hsub_half(a, b);
}

__device__ static inline __half __hsub_sat(__half a, __half b) {
  return __hip_hc_ir_hsub_half(a, b);
}

__device__ static inline __half hdiv(__half a, __half b) {
  return a/b;
}

#endif

#if __clang_major__ == 3

typedef struct {
  unsigned x: 16;
} __half;

typedef struct __attribute__((aligned(4))){
  union {
    __half p[2];
    unsigned int q;
  };
} __half2;




#endif


#endif
