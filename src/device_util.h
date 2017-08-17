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

#ifndef DEVICE_UTIL_H
#define DEVICE_UTIL_H

#include<hip/hcc_detail/hip_runtime.h>

/*
 Heap size computation for malloc and free device functions.
*/

#define NUM_PAGES_PER_THREAD  16
#define SIZE_OF_PAGE          64
#define NUM_THREADS_PER_CU    64
#define NUM_CUS_PER_GPU       64  // Specific for r9 Nano
#define NUM_PAGES NUM_PAGES_PER_THREAD * NUM_THREADS_PER_CU * NUM_CUS_PER_GPU
#define SIZE_MALLOC NUM_PAGES * SIZE_OF_PAGE
#define SIZE_OF_HEAP SIZE_MALLOC

#define HIP_SQRT_2 1.41421356237
#define HIP_SQRT_PI 1.77245385091

#define __hip_erfinva3 -0.140543331
#define __hip_erfinva2 0.914624893
#define __hip_erfinva1 -1.645349621
#define __hip_erfinva0 0.886226899

#define __hip_erfinvb4 0.012229801
#define __hip_erfinvb3 -0.329097515
#define __hip_erfinvb2 1.442710462
#define __hip_erfinvb1 -2.118377725
#define __hip_erfinvb0 1

#define __hip_erfinvc3 1.641345311
#define __hip_erfinvc2 3.429567803
#define __hip_erfinvc1 -1.62490649
#define __hip_erfinvc0 -1.970840454

#define __hip_erfinvd2 1.637067800
#define __hip_erfinvd1 3.543889200
#define __hip_erfinvd0 1

#define HIP_PI 3.14159265358979323846

__device__ void* __hip_hc_malloc(size_t size);
__device__ void* __hip_hc_free(void* ptr);

__device__ float __hip_erfinvf(float x);
__device__ double __hip_erfinv(double x);

__device__ float __hip_j0f(float x);
__device__ double __hip_j0(double x);

__device__ float __hip_j1f(float x);
__device__ double __hip_j1(double x);

__device__ float __hip_y0f(float x);
__device__ double __hip_y0(double x);

__device__ float __hip_y1f(float x);
__device__ double __hip_y1(double x);

__device__ float __hip_jnf(int n, float x);
__device__ double __hip_jn(int n, double x);

__device__ float __hip_ynf(int n, float x);
__device__ double __hip_yn(int n, double x);

__device__ float __hip_precise_cosf(float x);
__device__ float __hip_precise_exp10f(float x);
__device__ float __hip_precise_expf(float x);
__device__ float __hip_precise_frsqrt_rn(float x);
__device__ float __hip_precise_fsqrt_rd(float x);
__device__ float __hip_precise_fsqrt_rn(float x);
__device__ float __hip_precise_fsqrt_ru(float x);
__device__ float __hip_precise_fsqrt_rz(float x);
__device__ float __hip_precise_log10f(float x);
__device__ float __hip_precise_log2f(float x);
__device__ float __hip_precise_logf(float x);
__device__ float __hip_precise_powf(float base, float exponent);
__device__ void __hip_precise_sincosf(float x, float *s, float *c);
__device__ float __hip_precise_sinf(float x);
__device__ float __hip_precise_tanf(float x);
// Double Precision Math
__device__ double __hip_precise_dsqrt_rd(double x);
__device__ double __hip_precise_dsqrt_rn(double x);
__device__ double __hip_precise_dsqrt_ru(double x);
__device__ double __hip_precise_dsqrt_rz(double x);



// Float Fast Math
__device__ float __hip_fast_exp10f(float x);
__device__ float __hip_fast_expf(float x);
__device__ float __hip_fast_frsqrt_rn(float x);
__device__ float __hip_fast_fsqrt_rn(float x);
__device__ float __hip_fast_fsqrt_ru(float x);
__device__ float __hip_fast_fsqrt_rz(float x);
__device__ float __hip_fast_log10f(float x);
__device__ float __hip_fast_logf(float x);
__device__ float __hip_fast_powf(float base, float exponent);
__device__ void __hip_fast_sincosf(float x, float *s, float *c);
__device__ float __hip_fast_tanf(float x);
// Double Precision Math
__device__ double __hip_fast_dsqrt_rd(double x);
__device__ double __hip_fast_dsqrt_rn(double x);
__device__ double __hip_fast_dsqrt_ru(double x);
__device__ double __hip_fast_dsqrt_rz(double x);
__device__ void  __threadfence_system(void);

float __hip_host_j0f(float x);
double __hip_host_j0(double x);

float __hip_host_j1f(float x);
double __hip_host_j1(double x);

float __hip_host_y0f(float x);
double __hip_host_y1(double x);

float __hip_host_y1f(float x);
double __hip_host_y1(double x);

float __hip_host_jnf(int n, float x);
double __hip_host_jn(int n, double x);

float __hip_host_ynf(int n, float x);
double __hip_host_yn(int n, double x);

#endif
