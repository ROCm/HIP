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

#include <hc.hpp>
#include <grid_launch.h>
#include <hc_math.hpp>

#include "hip/hip_runtime.h"

//=================================================================================================
/*
 Implementation of malloc and free device functions.

 This is the best place to put them because the device
 global variables need to be initialized at the start.
*/

#define NUM_PAGES_PER_THREAD  16
#define SIZE_OF_PAGE          64
#define NUM_THREADS_PER_CU    64
#define NUM_CUS_PER_GPU       64
#define NUM_PAGES NUM_PAGES_PER_THREAD * NUM_THREADS_PER_CU * NUM_CUS_PER_GPU
#define SIZE_MALLOC NUM_PAGES * SIZE_OF_PAGE
#define SIZE_OF_HEAP SIZE_MALLOC

size_t g_malloc_heap_size = SIZE_OF_HEAP;

__attribute__((address_space(1))) char gpuHeap[SIZE_OF_HEAP];
__attribute__((address_space(1))) uint32_t gpuFlags[NUM_PAGES];

__device__ void *__hip_hc_malloc(size_t size)
{
    char *heap = (char*)gpuHeap;
    if(size > SIZE_OF_HEAP)
    {
        return (void*)nullptr;
    }
    uint32_t totalThreads = hipBlockDim_x * hipGridDim_x * hipBlockDim_y * hipGridDim_y * hipBlockDim_z * hipGridDim_z;
    uint32_t currentWorkItem = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    uint32_t numHeapsPerWorkItem = NUM_PAGES / totalThreads;
    uint32_t heapSizePerWorkItem = SIZE_OF_HEAP / totalThreads;

    uint32_t stride = size / SIZE_OF_PAGE;
    uint32_t start = numHeapsPerWorkItem * currentWorkItem;

    uint32_t k=0;

    while(gpuFlags[k] > 0)
    {
        k++;
    }

    for(uint32_t i=0;i<stride-1;i++)
    {
        gpuFlags[i+start+k] = 1;
    }

    gpuFlags[start+stride-1+k] = 2;

    void* ptr = (void*)(heap + heapSizePerWorkItem * currentWorkItem + k*SIZE_OF_PAGE);

    return ptr;
}

__device__ void* __hip_hc_free(void *ptr)
{
    if(ptr == nullptr)
    {
       return nullptr;
    }

    uint32_t offsetByte = (uint64_t)ptr - (uint64_t)gpuHeap;
    uint32_t offsetPage = offsetByte / SIZE_OF_PAGE;

    while(gpuFlags[offsetPage] != 0) {
        if(gpuFlags[offsetPage] == 2) {
            gpuFlags[offsetPage] = 0;
            offsetPage++;
            break;
        } else {
            gpuFlags[offsetPage] = 0;
            offsetPage++;
        }
    }

    return nullptr;
}

__device__ unsigned __hip_ds_bpermute(int index, unsigned src) {
    return hc::__amdgcn_ds_bpermute(index, src);
}

__device__ float __hip_ds_bpermutef(int index, float src) {
    return hc::__amdgcn_ds_bpermute(index, src);
}

__device__ unsigned __hip_ds_permute(int index, unsigned src) {
    return hc::__amdgcn_ds_permute(index, src);
}

__device__ float __hip_ds_permutef(int index, float src) {
    return hc::__amdgcn_ds_permute(index, src);
}

__device__ unsigned __hip_ds_swizzle(unsigned int src, int pattern) {
    return hc::__amdgcn_ds_swizzle(src, pattern);
}

__device__ float __hip_ds_swizzlef(float src, int pattern) {
    return hc::__amdgcn_ds_swizzle(src, pattern);
}

__device__ int __hip_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl) {
    return hc::__amdgcn_move_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}

#define MASK1 0x00ff00ff
#define MASK2 0xff00ff00

__device__ char4 __hip_hc_add8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.val & MASK1;
    unsigned one2 = in2.val & MASK1;
    out.val = (one1 + one2) & MASK1;
    one1 = in1.val & MASK2;
    one2 = in2.val & MASK2;
    out.val = out.val | ((one1 + one2) & MASK2);
    return out;
}

__device__ char4 __hip_hc_sub8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.val & MASK1;
    unsigned one2 = in2.val & MASK1;
    out.val = (one1 - one2) & MASK1;
    one1 = in1.val & MASK2;
    one2 = in2.val & MASK2;
    out.val = out.val | ((one1 - one2) & MASK2);
    return out;
}

__device__ char4 __hip_hc_mul8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.val & MASK1;
    unsigned one2 = in2.val & MASK1;
    out.val = (one1 * one2) & MASK1;
    one1 = in1.val & MASK2;
    one2 = in2.val & MASK2;
    out.val = out.val | ((one1 * one2) & MASK2);
    return out;
}

// loop unrolling
__device__ void* memcpy(void* dst, void* src, size_t size)
{
    uint8_t *dstPtr, *srcPtr;
    dstPtr = (uint8_t*)dst;
    srcPtr = (uint8_t*)src;
    for(uint32_t i=0;i<size;i++) {
        dstPtr[i] = srcPtr[i];
    }
    return nullptr;
}

__device__ void* memset(void* ptr, uint8_t val, size_t size)
{
    uint8_t *dstPtr;
    dstPtr = (uint8_t*)ptr;
    for(uint32_t i=0;i<size;i++) {
        dstPtr[i] = val;
    }
    return nullptr;
}

__device__ void* malloc(size_t size)
{
    return __hip_hc_malloc(size);
}

__device__ void* free(void *ptr)
{
    return __hip_hc_free(ptr);
}

//=================================================================================================

// TODO: Choose whether default is precise math or fast math based on compilation flag.
#ifdef __HCC_ACCELERATOR__
using namespace hc::precise_math;
#endif


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

__device__ float __hip_erfinvf(float x){
    float ret;
    int  sign;
    if (x < -1 || x > 1){
        return NAN;
    }
    if (x == 0){
        return 0;
    }
    if (x > 0){
        sign = 1;
    } else {
        sign = -1;
        x = -x;
    }
    if (x <= 0.7) {
        float x1 = x * x;
        float x2 = __hip_erfinva3 * x1 + __hip_erfinva2;
        float x3 = x2 * x1 + __hip_erfinva1;
        float x4 = x * (x3 * x1 + __hip_erfinva0);

        float r1 = __hip_erfinvb4 * x1 + __hip_erfinvb3;
        float r2 = r1 * x1 + __hip_erfinvb2;
        float r3 = r2 * x1 + __hip_erfinvb1;
        ret = x4 / (r3 * x1 + __hip_erfinvb0);
    } else {
        float x1 = hc::precise_math::sqrtf(-hc::precise_math::logf((1 - x) / 2));
        float x2 = __hip_erfinvc3 * x1 + __hip_erfinvc2;
        float x3 = x2 * x1 + __hip_erfinvc1;
        float x4 = x3 * x1 + __hip_erfinvc0;

        float r1 = __hip_erfinvd2 * x1 + __hip_erfinvd1;
        ret = x4 / (r1 * x1 + __hip_erfinvd0);
    }

    ret = ret * sign;
    x = x * sign;

    ret -= (hc::precise_math::erff(ret) - x) / (2 / HIP_SQRT_PI * hc::precise_math::expf(-ret * ret));
    ret -= (hc::precise_math::erff(ret) - x) / (2 / HIP_SQRT_PI * hc::precise_math::expf(-ret * ret));

    return ret;
}

__device__ double __hip_erfinv(double x){
    double ret;
    int  sign;
    if (x < -1 || x > 1){
        return NAN;
    }
    if (x == 0){
        return 0;
    }
    if (x > 0){
        sign = 1;
    } else {
        sign = -1;
        x = -x;
    }
    if (x <= 0.7) {
        double x1 = x * x;
        double x2 = __hip_erfinva3 * x1 + __hip_erfinva2;
        double x3 = x2 * x1 + __hip_erfinva1;
        double x4 = x * (x3 * x1 + __hip_erfinva0);

        double r1 = __hip_erfinvb4 * x1 + __hip_erfinvb3;
        double r2 = r1 * x1 + __hip_erfinvb2;
        double r3 = r2 * x1 + __hip_erfinvb1;
        ret = x4 / (r3 * x1 + __hip_erfinvb0);
    } else {
        double x1 = hc::precise_math::sqrt(-hc::precise_math::log((1 - x) / 2));
        double x2 = __hip_erfinvc3 * x1 + __hip_erfinvc2;
        double x3 = x2 * x1 + __hip_erfinvc1;
        double x4 = x3 * x1 + __hip_erfinvc0;

        double r1 = __hip_erfinvd2 * x1 + __hip_erfinvd1;
        ret = x4 / (r1 * x1 + __hip_erfinvd0);
    }

    ret = ret * sign;
    x = x * sign;

    ret -= (hc::precise_math::erf(ret) - x) / (2 / HIP_SQRT_PI * hc::precise_math::exp(-ret * ret));
    ret -= (hc::precise_math::erf(ret) - x) / (2 / HIP_SQRT_PI * hc::precise_math::exp(-ret * ret));

    return ret;
}

#define __hip_j0a1 57568490574.0
#define __hip_j0a2 -13362590354.0
#define __hip_j0a3 651619640.7
#define __hip_j0a4 -11214424.18
#define __hip_j0a5 77392.33017
#define __hip_j0a6 -184.9052456

#define __hip_j0b1 57568490411.0
#define __hip_j0b2 1029532985.0
#define __hip_j0b3 9494680.718
#define __hip_j0b4 59272.64853
#define __hip_j0b5 267.8532712

#define __hip_j0c 0.785398164
#define __hip_j0c1 -0.1098628627e-2
#define __hip_j0c2 0.2734510407e-4
#define __hip_j0c3 -0.2073370639e-5
#define __hip_j0c4 0.2093887211e-6

#define __hip_j0d1 -0.1562499995e-1
#define __hip_j0d2 0.1430488765e-3
#define __hip_j0d3 0.6911147651e-5
#define __hip_j0d4 0.7621095161e-6
#define __hip_j0d5 0.934935152e-7

#define __hip_j0e 0.636619772

__device__ double __hip_j0(double x)
{
    double ret, a = hc::precise_math::fabs(x);
    if (a < 8.0) {
        double y = x*x;
        double y1 = __hip_j0a6 * y + __hip_j0a5;
        double z1 = 1.0 * y + __hip_j0b5;

        double y2 = y1 * y + __hip_j0a4;
        double z2 = z1 * y + __hip_j0b4;

        double y3 = y2 * y + __hip_j0a3;
        double z3 = z2 * y + __hip_j0b3;

        double y4 = y3 * y + __hip_j0a2;
        double z4 = z3 * y + __hip_j0b2;

        double y5 = y4 * y + __hip_j0a1;
        double z5 = z4 * y + __hip_j0b1;

        ret = y5 / z5;

    }
    else {
        double z = 8.0 / a;
        double y = z*z;
        double x1 = a - __hip_j0c;

        double y1 = __hip_j0c4 * y + __hip_j0c3;
        double z1 = __hip_j0d5 * y + __hip_j0d4;

        double y2 = y1 * y + __hip_j0c2;
        double z2 = z1 * z + __hip_j0d3;

        double y3 = y2 * y + __hip_j0c1;
        double z3 = z2 * y + __hip_j0d2;

        double y4 = y3 * y + 1.0;
        double z4 = z3 * y + __hip_j0d1;

        ret = hc::precise_math::sqrt(__hip_j0e / a)*(hc::precise_math::cos(x1) * y4 - z * hc::precise_math::sin(x1) * z4);
    }
    return ret;
}

__device__ float __hip_j0f(float x)
{
    float ret, a = hc::precise_math::fabsf(x);
    if (a < 8.0) {
        float y = x*x;
        float y1 = __hip_j0a6 * y + __hip_j0a5;
        float z1 = 1.0 * y + __hip_j0b5;

        float y2 = y1 * y + __hip_j0a4;
        float z2 = z1 * y + __hip_j0b4;

        float y3 = y2 * y + __hip_j0a3;
        float z3 = z2 * y + __hip_j0b3;

        float y4 = y3 * y + __hip_j0a2;
        float z4 = z3 * y + __hip_j0b2;

        float y5 = y4 * y + __hip_j0a1;
        float z5 = z4 * y + __hip_j0b1;

        ret = y5 / z5;

    }
    else {
        float z = 8.0 / a;
        float y = z*z;
        float x1 = a - __hip_j0c;

        float y1 = __hip_j0c4 * y + __hip_j0c3;
        float z1 = __hip_j0d5 * y + __hip_j0d4;

        float y2 = y1 * y + __hip_j0c2;
        float z2 = z1 * z + __hip_j0d3;

        float y3 = y2 * y + __hip_j0c1;
        float z3 = z2 * y + __hip_j0d2;

        float y4 = y3 * y + 1.0;
        float z4 = z3 * y + __hip_j0d1;

        ret = hc::precise_math::sqrtf(__hip_j0e / a)*(hc::precise_math::cosf(x1) * y4 - z * hc::precise_math::sinf(x1) * z4);
    }
    return ret;
}

#define __hip_j1a1 -30.16036606
#define __hip_j1a2 15704.48260
#define __hip_j1a3 -2972611.439
#define __hip_j1a4 242396853.1
#define __hip_j1a5 -7895059235.0
#define __hip_j1a6 72362614232.0

#define __hip_j1b1 376.9991397
#define __hip_j1b2 99447.43394
#define __hip_j1b3 18583304.74
#define __hip_j1b4 2300535178.0
#define __hip_j1b5 144725228442.0

#define __hip_j1c 2.356194491
#define __hip_j1c1 -0.240337019e-6
#define __hip_j1c2 0.2457520174e-5
#define __hip_j1c3 -0.3516396496e-4
#define __hip_j1c4 0.183105e-2

#define __hip_j1d1 0.105787412e-6
#define __hip_j1d2 -0.88228987e-6
#define __hip_j1d3 0.8449199096e-5
#define __hip_j1d4 -0.2002690873e-3
#define __hip_j1d5 0.04687499995

#define __hip_j1e 0.636619772

__device__ double __hip_j1(double x)
{
    double ret, a = hc::precise_math::fabs(x);
    if (a < 8.0) {
        double y = x*x;

        double y1 = __hip_j1a1 * y + __hip_j1a2;
        double z1 = 1.0 * y + __hip_j1b1;

        double y2 = y1 * y + __hip_j1a3;
        double z2 = z1 * y + __hip_j1b2;

        double y3 = y2 * y + __hip_j1a4;
        double z3 = z2 * y + __hip_j1b3;

        double y4 = y3 * y + __hip_j1a5;
        double z4 = z3 * y + __hip_j1b4;

        double y5 = y4 * y + __hip_j1a6;
        double z5 = z4 * y + __hip_j1b5;

        ret = x * y5 / z5;

    }
    else {
        double z = 8.0 / a;
        double y = z*z;
        double x1 = a - __hip_j1c;

        double y1 = __hip_j1c1 * y + __hip_j1c2;
        double y2 = y1 * y + __hip_j1c3;
        double y3 = y2 * y + __hip_j1c4;
        double y4 = y3 * y + 1.0;

        double z1 = __hip_j1d1 * y + __hip_j1d2;
        double z2 = z1 * y + __hip_j1d3;
        double z3 = z2 * y + __hip_j1d4;
        double z4 = z3 * y + __hip_j1d5;

        ret = hc::precise_math::sqrt(__hip_j1e / a)*(hc::precise_math::cos(x1)*y4 - z*hc::precise_math::sin(x1)*z4);
        if (x < 0.0) ret = -ret;
    }
    return ret;
}

__device__ float __hip_j1f(float x)
{
    double ret, a = hc::precise_math::fabsf(x);
    if (a < 8.0) {
        float y = x*x;

        float y1 = __hip_j1a1 * y + __hip_j1a2;
        float z1 = 1.0 * y + __hip_j1b1;

        float y2 = y1 * y + __hip_j1a3;
        float z2 = z1 * y + __hip_j1b2;

        float y3 = y2 * y + __hip_j1a4;
        float z3 = z2 * y + __hip_j1b3;

        float y4 = y3 * y + __hip_j1a5;
        float z4 = z3 * y + __hip_j1b4;

        float y5 = y4 * y + __hip_j1a6;
        float z5 = z4 * y + __hip_j1b5;

        ret = x * y5 / z5;

    }
    else {
        float z = 8.0 / a;
        float y = z*z;
        float x1 = a - __hip_j1c;

        float y1 = __hip_j1c1 * y + __hip_j1c2;
        float y2 = y1 * y + __hip_j1c3;
        float y3 = y2 * y + __hip_j1c4;
        float y4 = y3 * y + 1.0;

        float z1 = __hip_j1d1 * y + __hip_j1d2;
        float z2 = z1 * y + __hip_j1d3;
        float z3 = z2 * y + __hip_j1d4;
        float z4 = z3 * y + __hip_j1d5;

        ret = hc::precise_math::sqrtf(__hip_j1e / a)*(hc::precise_math::cosf(x1)*y4 - z*hc::precise_math::sinf(x1)*z4);
        if (x < 0.0) ret = -ret;
    }
    return ret;
}

#define __hip_y0a1 228.4622733
#define __hip_y0a2 -86327.92757
#define __hip_y0a3 10879881.29
#define __hip_y0a4 -512359803.6
#define __hip_y0a5 7062834065.0
#define __hip_y0a6 -2957821389.0

#define __hip_y0b1 226.1030244
#define __hip_y0b2 47447.26470
#define __hip_y0b3 7189466.438
#define __hip_y0b4 745249964.8
#define __hip_y0b5 40076544269.0

#define __hip_y0c 0.636619772

#define __hip_y0d 0.785398164

#define __hip_y0e1 0.2093887211e-6
#define __hip_y0e2 -0.2073370639e-5
#define __hip_y0e3 0.2734510407e-4
#define __hip_y0e4 -0.1098628627e-2


#define __hip_y0f1 -0.934945152e-7
#define __hip_y0f2 0.7621095161e-6
#define __hip_y0f3 -0.6911147651e-5
#define __hip_y0f4 0.1430488765e-3
#define __hip_y0f5 -0.1562499995e-1

#define __hip_y1g 0.636619772

__device__ double __hip_y0(double x)
{
    double ret;

    if (x < 8.0) {
        double y = x*x;
        double y1 = __hip_y0a1 * y + __hip_y0a2;
        double y2 = y1 * y + __hip_y0a3;
        double y3 = y2 * y + __hip_y0a4;
        double y4 = y3 * y + __hip_y0a5;
        double y5 = y4 * y + __hip_y0a6;

        double z1 = 1.0 * y + __hip_y0b1;
        double z2 = z1 * y + __hip_y0b2;
        double z3 = z2 * y + __hip_y0b3;
        double z4 = z3 * y + __hip_y0b4;
        double z5 = z4 * y + __hip_y0b5;


        ret = (y5 / z5) + __hip_y0c * __hip_j0(x) * hc::precise_math::log(x);
    }
    else {
        double z = 8.0 / x;
        double y = z*z;
        double x1 = x - __hip_y0d;

        double y1 = __hip_y0e1 * y + __hip_y0e2;
        double y2 = y1 * y + __hip_y0e3;
        double y3 = y2 * y + __hip_y0e4;
        double y4 = y3 * y + 1.0;

        double z1 = __hip_y0f1 * y + __hip_y0f2;
        double z2 = z1 * y + __hip_y0f3;
        double z3 = z2 * y + __hip_y0f4;
        double z4 = z3 * y + __hip_y0f5;

        ret = hc::precise_math::sqrt(__hip_y1g / x)*(hc::precise_math::sin(x1)*y4 + z * hc::precise_math::cos(x1) * z4);
    }
    return ret;

}


__device__ float __hip_y0f(float x)
{
    float ret;

    if (x < 8.0) {
        float y = x*x;
        float y1 = __hip_y0a1 * y + __hip_y0a2;
        float y2 = y1 * y + __hip_y0a3;
        float y3 = y2 * y + __hip_y0a4;
        float y4 = y3 * y + __hip_y0a5;
        float y5 = y4 * y + __hip_y0a6;

        float z1 = 1.0 * y + __hip_y0b1;
        float z2 = z1 * y + __hip_y0b2;
        float z3 = z2 * y + __hip_y0b3;
        float z4 = z3 * y + __hip_y0b4;
        float z5 = z4 * y + __hip_y0b5;


        ret = (y5 / z5) + __hip_y0c * __hip_j0f(x) * hc::precise_math::logf(x);
    }
    else {
        float z = 8.0 / x;
        float y = z*z;
        float x1 = x - __hip_y0d;

        float y1 = __hip_y0e1 * y + __hip_y0e2;
        float y2 = y1 * y + __hip_y0e3;
        float y3 = y2 * y + __hip_y0e4;
        float y4 = y3 * y + 1.0;

        float z1 = __hip_y0f1 * y + __hip_y0f2;
        float z2 = z1 * y + __hip_y0f3;
        float z3 = z2 * y + __hip_y0f4;
        float z4 = z3 * y + __hip_y0f5;

        ret = hc::precise_math::sqrtf(__hip_y1g / x)*(hc::precise_math::sinf(x1)*y4 + z * hc::precise_math::cosf(x1) * z4);
    }
    return ret;

}

#define __hip_y1a1 0.8511937935e4
#define __hip_y1a2 -0.4237922726e7
#define __hip_y1a3 0.7349264551e9
#define __hip_y1a4 -0.5153438139e11
#define __hip_y1a5 0.1275274390e13
#define __hip_y1a6 -0.4900604943e13

#define __hip_y1b1 0.3549632885e3
#define __hip_y1b2 0.1020426050e6
#define __hip_y1b3 0.2245904002e8
#define __hip_y1b4 0.3733650367e10
#define __hip_y1b5 0.4244419664e12
#define __hip_y1b6 0.2499580570e14

#define __hip_y1c 0.636619772

#define __hip_y1d 2.356194491

#define __hip_y1e1 -0.240337019e-6
#define __hip_y1e2 0.2457520174e-5
#define __hip_y1e3 -0.3516396496e-4
#define __hip_y1e4 0.183105e-2

#define __hip_y1f1 0.105787412e-6
#define __hip_y1f2 -0.88228987e-6
#define __hip_y1f3 0.8449199096e-5
#define __hip_y1f4 -0.2002690873e-3
#define __hip_y1f5 0.04687499995

#define __hip_y1g 0.636619772

__device__ double __hip_y1(double x)
{
    double ret;

    if (x < 8.0) {
        double y = x*x;

        double y1 = __hip_y1a1 * y + __hip_y1a2;
        double y2 = y1 * y + __hip_y1a3;
        double y3 = y2 * y + __hip_y1a4;
        double y4 = y3 * y + __hip_y1a5;
        double y5 = y4 * y + __hip_y1a6;
        double y6 = x * y5;

        double z1 = __hip_y1b1 + y;
        double z2 = z1 * y + __hip_y1b2;
        double z3 = z2 * y + __hip_y1b3;
        double z4 = z3 * y + __hip_y1b4;
        double z5 = z4 * y + __hip_y1b5;
        double z6 = z5 * y + __hip_y1b6;

        ret = (y6 / z6) + __hip_y1c * (__hip_j1(x) * hc::precise_math::log(x) - 1.0 / x);
    }
    else {
        double z = 8.0 / x;
        double y = z*z;
        double x1 = x - __hip_y1d;

        double y1 = __hip_y1e1 * y + __hip_y1e2;
        double y2 = y1 * y + __hip_y1e3;
        double y3 = y2 * y + __hip_y1e4;
        double y4 = y3 * y + 1.0;

        double z1 = __hip_y1f1 * y + __hip_y1f2;
        double z2 = z1 * y + __hip_y1f3;
        double z3 = z2 * y + __hip_y1f4;
        double z4 = z3 * y + __hip_y1f5;

        ret = hc::precise_math::sqrt(__hip_y1g / x)*(hc::precise_math::sin(x1)*y4 + z*hc::precise_math::cos(x1)*z4);
    }
    return ret;
}

__device__ float __hip_y1f(float x)
{
    float ret;

    if (x < 8.0) {
        float y = x*x;

        float y1 = __hip_y1a1 * y + __hip_y1a2;
        float y2 = y1 * y + __hip_y1a3;
        float y3 = y2 * y + __hip_y1a4;
        float y4 = y3 * y + __hip_y1a5;
        float y5 = y4 * y + __hip_y1a6;
        float y6 = x * y5;

        float z1 = __hip_y1b1 + y;
        float z2 = z1 * y + __hip_y1b2;
        float z3 = z2 * y + __hip_y1b3;
        float z4 = z3 * y + __hip_y1b4;
        float z5 = z4 * y + __hip_y1b5;
        float z6 = z5 * y + __hip_y1b6;

        ret = (y6 / z6) + __hip_y1c * (__hip_j1f(x) * hc::precise_math::logf(x) - 1.0 / x);
    }
    else {
        float z = 8.0 / x;
        float y = z*z;
        float x1 = x - __hip_y1d;

        float y1 = __hip_y1e1 * y + __hip_y1e2;
        float y2 = y1 * y + __hip_y1e3;
        float y3 = y2 * y + __hip_y1e4;
        float y4 = y3 * y + 1.0;

        float z1 = __hip_y1f1 * y + __hip_y1f2;
        float z2 = z1 * y + __hip_y1f3;
        float z3 = z2 * y + __hip_y1f4;
        float z4 = z3 * y + __hip_y1f5;

        ret = hc::precise_math::sqrtf(__hip_y1g / x)*(hc::precise_math::sinf(x1)*y4 + z*hc::precise_math::cosf(x1)*z4);
    }
    return ret;
}

#define __hip_k1 40.0
#define __hip_k2 1.0e10
#define __hip_k3 1.0e-10

__device__ double __hip_jn(int n, double x)
{
    int    sum = 0, m;
    double a, b0, b1, b2, val, t, ret;
    if (n < 0){
        return NAN;
    }
    a = hc::precise_math::fabs(x);
    if (n == 0){
        return(__hip_j0(a));
    }
    if (n == 1){
        return(__hip_j1(a));
    }
    if (a == 0.0){
        return 0.0;
    }
    else if (a > (double)n) {
        t = 2.0 / a;
        b1 = __hip_j0(a);
        b0 = __hip_j1(a);
        for (int i = 1; i<n; i++) {
            b2 = i*t*b0 - b1;
            b1 = b0;
            b0 = b2;
        }
        ret = b0;
    }
    else {
        t = 2.0 / a;
        m = 2 * ((n + (int)hc::precise_math::sqrt(__hip_k1*n)) / 2);
        b2 = ret = val = 0.0;
        b0 = 1.0;
        for (int i = m; i>0; i--) {
            b1 = i*t*b0 - b2;
            b2 = b0;
            b0 = b1;
            if (hc::precise_math::fabs(b0) > __hip_k2) {
                b0 *= __hip_k3;
                b2 *= __hip_k3;
                ret *= __hip_k3;
                val *= __hip_k3;
            }
            if (sum) val += b0;
            sum = !sum;
            if (i == n) ret = b2;
        }
        val = 2.0*val - b0;
        ret /= val;
    }
    return  x < 0.0 && n % 2 == 1 ? -ret : ret;
}

__device__ float __hip_jnf(int n, float x)
{
    int    sum = 0, m;
    float a, b0, b1, b2, val, t, ret;
    if (n < 0){
        return NAN;
    }
    a = hc::precise_math::fabsf(x);
    if (n == 0){
        return(__hip_j0f(a));
    }
    if (n == 1){
        return(__hip_j1f(a));
    }
    if (a == 0.0){
        return 0.0;
    }
    else if (a > (float)n) {
        t = 2.0 / a;
        b1 = __hip_j0f(a);
        b0 = __hip_j1f(a);
        for (int i = 1; i<n; i++) {
            b2 = i*t*b0 - b1;
            b1 = b0;
            b0 = b2;
        }
        ret = b0;
    }
    else {
        t = 2.0 / a;
        m = 2 * ((n + (int)hc::precise_math::sqrtf(__hip_k1*n)) / 2);
        b2 = ret = val = 0.0;
        b0 = 1.0;
        for (int i = m; i>0; i--) {
            b1 = i*t*b0 - b2;
            b2 = b0;
            b0 = b1;
            if (hc::precise_math::fabsf(b0) > __hip_k2) {
                b0 *= __hip_k3;
                b2 *= __hip_k3;
                ret *= __hip_k3;
                val *= __hip_k3;
            }
            if (sum) val += b0;
            sum = !sum;
            if (i == n) ret = b2;
        }
        val = 2.0*val - b0;
        ret /= val;
    }
    return  x < 0.0 && n % 2 == 1 ? -ret : ret;
}

__device__ double __hip_yn(int n, double x)
{
    double b0, b1, b2, t;

    if (n < 0 || x == 0.0)
    {
        return NAN;
    }
    if (n == 0){
        return(__hip_y0(x));
    }
    if (n == 1){
        return(__hip_y1(x));
    }
    t = 2.0 / x;
    b0 = __hip_y1(x);
    b1 = __hip_y0(x);
    for (int i = 1; i<n; i++) {
        b2 = i*t*b0 - b1;
        b1 = b0;
        b0 = b2;
    }
    return b0;
}

__device__ float __hip_ynf(int n, float x)
{
    float b0, b1, b2, t;

    if (n < 0 || x == 0.0)
    {
        return NAN;
    }
    if (n == 0){
        return(__hip_y0f(x));
    }
    if (n == 1){
        return(__hip_y1f(x));
    }
    t = 2.0 / x;
    b0 = __hip_y1f(x);
    b1 = __hip_y0f(x);
    for (int i = 1; i<n; i++) {
        b2 = i*t*b0 - b1;
        b1 = b0;
        b0 = b2;
    }
    return b0;
}



__device__ float acosf(float x)
{
    return hc::precise_math::acosf(x);
}
__device__ float acoshf(float x)
{
    return hc::precise_math::acoshf(x);
}
__device__ float asinf(float x)
{
    return hc::precise_math::asinf(x);
}
__device__ float asinhf(float x)
{
    return hc::precise_math::asinhf(x);
}
__device__ float atan2f(float y, float x)
{
    return hc::precise_math::atan2f(x, y);
}
__device__ float atanf(float x)
{
    return hc::precise_math::atanf(x);
}
__device__ float atanhf(float x)
{
    return hc::precise_math::atanhf(x);
}
__device__ float cbrtf(float x)
{
    return hc::precise_math::cbrtf(x);
}
__device__ float ceilf(float x)
{
    return hc::precise_math::ceilf(x);
}
__device__ float copysignf(float x, float y)
{
    return hc::precise_math::copysignf(x, y);
}
__device__ float cosf(float x)
{
    return hc::precise_math::cosf(x);
}
__device__ float coshf(float x)
{
    return hc::precise_math::coshf(x);
}
__device__ float cyl_bessel_i0f(float x);
__device__ float cyl_bessel_i1f(float x);
__device__ float erfcf(float x)
{
    return hc::precise_math::erfcf(x);
}
__device__ float erfcinvf(float y)
{
    return __hip_erfinvf(1 - y);
}
__device__ float erfcxf(float x)
{
    return hc::precise_math::expf(x*x)*hc::precise_math::erfcf(x);
}
__device__ float erff(float x)
{
    return hc::precise_math::erff(x);
}
__device__ float erfinvf(float y)
{
    return __hip_erfinvf(y);
}
__device__ float exp10f(float x)
{
    return hc::precise_math::exp10f(x);
}
__device__ float exp2f(float x)
{
    return hc::precise_math::exp2f(x);
}
__device__ float expf(float x)
{
    return hc::precise_math::expf(x);
}
__device__ float expm1f(float x)
{
    return hc::precise_math::expm1f(x);
}
__device__ float fabsf(float x)
{
    return hc::precise_math::fabsf(x);
}
__device__ float fdimf(float x, float y)
{
    return hc::precise_math::fdimf(x, y);
}
__device__ float fdividef(float x, float y)
{
    return x/y;
}
__device__ float floorf(float x)
{
    return hc::precise_math::floorf(x);
}
__device__ float fmaf(float x, float y, float z)
{
    return hc::precise_math::fmaf(x, y, z);
}
__device__ float fmaxf(float x, float y)
{
    return hc::precise_math::fmaxf(x, y);
}
__device__ float fminf(float x, float y)
{
    return hc::precise_math::fminf(x, y);
}
__device__ float fmodf(float x, float y)
{
    return hc::precise_math::fmodf(x, y);
}
__device__ float frexpf(float x, int *nptr)
{
    return hc::precise_math::frexpf(x, nptr);
}
__device__ float hypotf(float x, float y)
{
    return hc::precise_math::hypotf(x, y);
}
__device__ float ilogbf(float x)
{
    return hc::precise_math::ilogbf(x);
}
__device__ unsigned isfinite(float a)
{
    return hc::precise_math::isfinite(a);
}
__device__ unsigned isinf(float a)
{
    return hc::precise_math::isinf(a);
}
__device__ unsigned isnan(float a)
{
    return hc::precise_math::isnan(a);
}
__device__ float j0f(float x)
{
    return __hip_j0f(x);
}
__device__ float j1f(float x)
{
    return __hip_j1f(x);
}
__device__ float jnf(int n, float x)
{
    return __hip_jnf(n, x);
}
__device__ float ldexpf(float x, int exp)
{
    return hc::precise_math::ldexpf(x, exp);
}
__device__ float lgammaf(float x, int *sign)
{
    return hc::precise_math::lgammaf(x, sign);
}
__device__ long long int llrintf(float x)
{
    int y = hc::precise_math::roundf(x);
    long long int z = y;
    return z;
}
__device__ long long int llroundf(float x)
{
    int y = hc::precise_math::roundf(x);
    long long int z = y;
    return z;
}__device__ float log10f(float x)
{
    return hc::precise_math::log10f(x);
}
__device__ float log1pf(float x)
{
    return hc::precise_math::log1pf(x);
}
__device__ float log2f(float x)
{
    return hc::precise_math::log2f(x);
}
__device__ float logbf(float x)
{
    return hc::precise_math::logbf(x);
}
__device__ float logf(float x)
{
    return hc::precise_math::logf(x);
}
__device__ long int lrintf(float x)
{
    int y = hc::precise_math::roundf(x);
    long int z = y;
    return z;
}
__device__ long int lroundf(float x)
{
    long int y = hc::precise_math::roundf(x);
    return y;
}
__device__ float modff(float x, float *iptr)
{
    return hc::precise_math::modff(x, iptr);
}
__device__ float nanf(const char* tagp)
{
    return hc::precise_math::nanf((int)*tagp);
}
__device__ float nearbyintf(float x)
{
    return hc::precise_math::nearbyintf(x);
}
__device__ float nextafterf(float x, float y)
{
    return hc::precise_math::nextafter(x, y);
}
__device__ float norm3df(float a, float b, float c)
{
     float x = a*a + b*b + c*c;
     return hc::precise_math::sqrtf(x);
}
__device__ float norm4df(float a, float b, float c, float d)
{
     float x = a*a + b*b;
     float y = c*c + d*d;
     return hc::precise_math::sqrtf(x+y);
}

__device__ float normcdff(float y)
{
     return ((hc::precise_math::erff(y)/1.41421356237) + 1)/2;
}
__device__ float normcdfinvf(float y)
{
     return HIP_SQRT_2 * __hip_erfinvf(2*y-1);
}
__device__ float normf(int dim, const float *a)
{
    float x = 0.0f;
    for(int i=0;i<dim;i++)
    {
        x = hc::precise_math::fmaf(a[i], a[i], x);
    }
    return hc::precise_math::sqrtf(x);
}
__device__ float powf(float x, float y)
{
    return hc::precise_math::powf(x, y);
}
__device__ float rcbrtf(float x)
{
    return hc::precise_math::rcbrtf(x);
}
__device__ float remainderf(float x, float y)
{
    return hc::precise_math::remainderf(x, y);
}
__device__ float remquof(float x, float y, int *quo)
{
    return hc::precise_math::remquof(x, y, quo);
}
__device__ float rhypotf(float x, float y)
{
    return 1/hc::precise_math::hypotf(x, y);
}
__device__ float rintf(float x)
{
    return hc::precise_math::roundf(x);
}
__device__ float rnorm3df(float a, float b, float c)
{
    float x = a*a + b*b + c*c;
    return 1/hc::precise_math::sqrtf(x);
}
__device__ float rnorm4df(float a, float b, float c, float d)
{
    float x = a*a + b*b;
    float y = c*c + d*d;
    return 1/hc::precise_math::sqrtf(x+y);
}
__device__ float rnormf(int dim, const float* a)
{
    float x = 0.0f;
    for(int i=0;i<dim;i++)
    {
        x = hc::precise_math::fmaf(a[i], a[i], x);
    }
    return 1/hc::precise_math::sqrtf(x);
}
__device__ float roundf(float x)
{
    return hc::precise_math::roundf(x);
}
__device__ float scalblnf(float x, long int n)
{
    return hc::precise_math::scalb(x, n);
}
__device__ float scalbnf(float x, int n)
{
    return hc::precise_math::scalbnf(x, n);
}
__device__ unsigned signbit(float a)
{
    return hc::precise_math::signbit(a);
}
__device__ void sincosf(float x, float *sptr, float *cptr)
{
    *sptr = hc::precise_math::sinf(x);
    *cptr = hc::precise_math::cosf(x);
}
__device__ void sincospif(float x, float *sptr, float *cptr)
{
    *sptr = hc::precise_math::sinpif(x);
    *cptr = hc::precise_math::cospif(x);
}
__device__ float sinf(float x)
{
    return hc::precise_math::sinf(x);
}
__device__ float sinhf(float x)
{
    return hc::precise_math::sinhf(x);
}
__device__ float tanf(float x)
{
    return hc::precise_math::tanf(x);
}
__device__ float tanhf(float x)
{
    return hc::precise_math::tanhf(x);
}
__device__ float tgammaf(float x)
{
    return hc::precise_math::tgammaf(x);
}
__device__ float truncf(float x)
{
    return hc::precise_math::truncf(x);
}
__device__ float y0f(float x)
{
    return __hip_y0f(x);
}
__device__ float y1f(float x)
{
    return __hip_y1f(x);
}
__device__ float ynf(int n, float x)
{
    return __hip_ynf(n, x);
}
__device__ float cospif(float x)
{
    return hc::precise_math::cospif(x);
}
__device__ float sinpif(float x)
{
    return hc::precise_math::sinpif(x);
}
__device__ float sqrtf(float x)
{
    return hc::precise_math::sqrtf(x);
}
__device__ float rsqrtf(float x)
{
    return hc::precise_math::rsqrtf(x);
}

/*
 * Double precision device math functions
 */

__device__ double acos(double x)
{
    return hc::precise_math::acos(x);
}
__device__ double acosh(double x)
{
    return hc::precise_math::acosh(x);
}
__device__ double asin(double x)
{
    return hc::precise_math::asin(x);
}
__device__ double asinh(double x)
{
    return hc::precise_math::asinh(x);
}
__device__ double atan(double x)
{
    return hc::precise_math::atan(x);
}
__device__ double atan2(double y, double x)
{
    return hc::precise_math::atan2(y, x);
}
__device__ double atanh(double x)
{
    return hc::precise_math::atanh(x);
}
__device__ double cbrt(double x)
{
    return hc::precise_math::cbrt(x);
}
__device__ double ceil(double x)
{
    return hc::precise_math::ceil(x);
}
__device__ double copysign(double x, double y)
{
    return hc::precise_math::copysign(x, y);
}
__device__ double cos(double x)
{
    return hc::precise_math::cos(x);
}
__device__ double cosh(double x)
{
    return hc::precise_math::cosh(x);
}
__device__ double cospi(double x)
{
    return hc::precise_math::cospi(x);
}
__device__ double cyl_bessel_i0(double x);
__device__ double cyl_bessel_i1(double x);
__device__ double erf(double x)
{
    return hc::precise_math::erf(x);
}
__device__ double erfc(double x)
{
    return hc::precise_math::erfc(x);
}
__device__ double erfcinv(double x)
{
    return __hip_erfinv(1 - x);
}
__device__ double erfcx(double x)
{
    return hc::precise_math::exp(x*x)*hc::precise_math::erf(x);
}
__device__ double erfinv(double x)
{
    return __hip_erfinv(x);
}
__device__ double exp(double x)
{
    return hc::precise_math::exp(x);
}
__device__ double exp10(double x)
{
    return hc::precise_math::exp10(x);
}
__device__ double exp2(double x)
{
    return hc::precise_math::exp2(x);
}
__device__ double expm1(double x)
{
    return hc::precise_math::expm1(x);
}
__device__ double fabs(double x)
{
    return hc::precise_math::fabs(x);
}
__device__ double fdim(double x, double y)
{
    return hc::precise_math::fdim(x, y);
}
__device__ double fdivide(double x, double y)
{
    return x/y;
}
__device__ double floor(double x)
{
    return hc::precise_math::floor(x);
}
__device__ double fma(double x, double y, double z)
{
    return hc::precise_math::fma(x, y, z);
}
__device__ double fmax(double x, double y)
{
    return hc::precise_math::fmax(x, y);
}
__device__ double fmin(double x, double y)
{
    return hc::precise_math::fmin(x, y);
}
__device__ double fmod(double x, double y)
{
    return hc::precise_math::fmod(x, y);
}
__device__ double frexp(double x, int *y)
{
    return hc::precise_math::frexp(x, y);
}
__device__ double hypot(double x, double y)
{
    return hc::precise_math::hypot(x, y);
}
__device__ double ilogb(double x)
{
    return hc::precise_math::ilogb(x);
}
__device__ unsigned isfinite(double x)
{
    return hc::precise_math::isfinite(x);
}
__device__ unsigned isinf(double x)
{
    return hc::precise_math::isinf(x);
}
__device__ unsigned isnan(double x)
{
    return hc::precise_math::isnan(x);
}
__device__ double j0(double x)
{
    return __hip_j0(x);
}
__device__ double j1(double x)
{
    return __hip_j1(x);
}
__device__ double jn(int n, double x)
{
    return __hip_jn(n, x);
}
__device__ double ldexp(double x, int exp)
{
    return hc::precise_math::ldexp(x, exp);
}
__device__ double lgamma(double x, int *sign)
{
    return hc::precise_math::lgamma(x, sign);
}
__device__ long long int llrint(double x)
{
    long long int y = hc::precise_math::round(x);
    return y;
}
__device__ long long int llround(double x)
{
    long long int y = hc::precise_math::round(x);
    return y;
}
__device__ double log(double x)
{
    return hc::precise_math::log(x);
}
__device__ double log10(double x)
{
    return hc::precise_math::log10(x);
}
__device__ double log1p(double x)
{
    return hc::precise_math::log1p(x);
}
__device__ double log2(double x)
{
    return hc::precise_math::log2(x);
}
__device__ double logb(double x)
{
    return hc::precise_math::logb(x);
}
__device__ long int lrint(double x)
{
    long int y = hc::precise_math::round(x);
    return y;
}
__device__ long int lround(double x)
{
    long int y = hc::precise_math::round(x);
    return y;
}
__device__ double modf(double x, double *iptr)
{
    return hc::precise_math::modf(x, iptr);
}
__device__ double nan(const char *tagp)
{
    return hc::precise_math::nan((int)*tagp);
}
__device__ double nearbyint(double x)
{
    return hc::precise_math::nearbyint(x);
}
__device__ double nextafter(double x, double y)
{
    return hc::precise_math::nextafter(x, y);
}
__device__ double norm3d(double a, double b, double c)
{
    double x = a*a + b*b + c*c;
    return hc::precise_math::sqrt(x);
}
__device__ double norm4d(double a, double b, double c, double d)
{
    double x = a*a + b*b;
    double y = c*c + d*d;
    return hc::precise_math::sqrt(x+y);
}
__device__ double normcdf(double y)
{
     return ((hc::precise_math::erf(y)/HIP_SQRT_2) + 1)/2;
}
__device__ double pow(double x, double y)
{
    return hc::precise_math::pow(x, y);
}
__device__ double rcbrt(double x)
{
    return hc::precise_math::rcbrt(x);
}
__device__ double remainder(double x, double y)
{
    return hc::precise_math::remainder(x, y);
}
__device__ double remquo(double x, double y, int *quo)
{
    return hc::precise_math::remquo(x, y, quo);
}
__device__ double rhypot(double x, double y)
{
    return 1/hc::precise_math::sqrt(x*x + y*y);
}
__device__ double rint(double x)
{
    return hc::precise_math::round(x);
}
__device__ double rnorm3d(double a, double b, double c)
{
    return hc::precise_math::rsqrt(a*a + b*b + c*c);
}
__device__ double rnorm4d(double a, double b, double c, double d)
{
    return hc::precise_math::rsqrt(a*a + b*b + c*c + d*d);
}
__device__ double rnorm(int dim, const double* t)
{
    double x = 0.0;
    for(int i=0;i<dim;i++)
    {
        x = hc::precise_math::fma(t[i], t[i], x);
    }
    return 1/x;
}
__device__ double round(double x)
{
    return hc::precise_math::round(x);
}
__device__ double rsqrt(double x)
{
    return hc::precise_math::rsqrt(x);
}
__device__ double scalbln(double x, long int n)
{
    return hc::precise_math::scalb(x, n);
}
__device__ double scalbn(double x, int n)
{
    return hc::precise_math::scalbn(x, n);
}
__device__ unsigned signbit(double x)
{
    return hc::precise_math::signbit(x);
}
__device__ double sin(double x)
{
    return hc::precise_math::sin(x);
}
__device__ void sincos(double x, double *sptr, double *cptr)
{
    *sptr = hc::precise_math::sin(x);
    *cptr = hc::precise_math::cos(x);
}
__device__ void sincospi(double x, double *sptr, double *cptr)
{
    *sptr = hc::precise_math::sinpi(x);
    *cptr = hc::precise_math::cospi(x);
}
__device__ double sinh(double x)
{
    return hc::precise_math::sinh(x);
}
__device__ double sinpi(double x)
{
    return hc::precise_math::sinpi(x);
}
__device__ double sqrt(double x)
{
    return hc::precise_math::sqrt(x);
}
__device__ double tan(double x)
{
    return hc::precise_math::tan(x);
}
__device__ double tanh(double x)
{
    return hc::precise_math::tanh(x);
}
__device__ double tgamma(double x)
{
    return hc::precise_math::tgamma(x);
}
__device__ double trunc(double x)
{
    return hc::precise_math::trunc(x);
}
__device__ double y0(double x)
{
    return __hip_y0(x);
}
__device__ double y1(double x)
{
    return __hip_y1(x);
}
__device__ double yn(int n, double x)
{
    return __hip_yn(n, x);
}

const int warpSize = 64;

__device__ long long int clock64() { return (long long int)hc::__cycle_u64(); };
__device__ clock_t clock() { return (clock_t)hc::__cycle_u64(); };


//atomicAdd()
__device__  int atomicAdd(int* address, int val)
{
	return hc::atomic_fetch_add(address,val);
}
__device__  unsigned int atomicAdd(unsigned int* address,
                       unsigned int val)
{
   return hc::atomic_fetch_add(address,val);
}
__device__  unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val)
{
 return (long long int)hc::atomic_fetch_add((uint64_t*)address,(uint64_t)val);
}
__device__  float atomicAdd(float* address, float val)
{
	return hc::atomic_fetch_add(address,val);
}

//atomicSub()
__device__  int atomicSub(int* address, int val)
{
	return hc::atomic_fetch_sub(address,val);
}
__device__  unsigned int atomicSub(unsigned int* address,
                       unsigned int val)
{
   return hc::atomic_fetch_sub(address,val);
}

//atomicExch()
__device__  int atomicExch(int* address, int val)
{
	return hc::atomic_exchange(address,val);
}
__device__  unsigned int atomicExch(unsigned int* address,
                        unsigned int val)
{
	return hc::atomic_exchange(address,val);
}
__device__  unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val)
{
	return (long long int)hc::atomic_exchange((uint64_t*)address,(uint64_t)val);
}
__device__  float atomicExch(float* address, float val)
{
	return hc::atomic_exchange(address,val);
}

//atomicMin()
__device__  int atomicMin(int* address, int val)
{
	return hc::atomic_fetch_min(address,val);
}
__device__  unsigned int atomicMin(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_min(address,val);
}
__device__  unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_min((uint64_t*)address,(uint64_t)val);
}

//atomicMax()
__device__  int atomicMax(int* address, int val)
{
	return hc::atomic_fetch_max(address,val);
}
__device__  unsigned int atomicMax(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_max(address,val);
}
__device__  unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_max((uint64_t*)address,(uint64_t)val);
}

//atomicCAS()
__device__  int atomicCAS(int* address, int compare, int val)
{
	hc::atomic_compare_exchange(address,&compare,val);
	return *address;
}
__device__  unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val)
{
	hc::atomic_compare_exchange(address,&compare,val);
	return *address;
}
__device__  unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val)
{
	hc::atomic_compare_exchange((uint64_t*)address,(uint64_t*)&compare,(uint64_t)val);
	return *address;
}

//atomicAnd()
__device__  int atomicAnd(int* address, int val)
{
	return hc::atomic_fetch_and(address,val);
}
__device__  unsigned int atomicAnd(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_and(address,val);
}
__device__  unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_and((uint64_t*)address,(uint64_t)val);
}

//atomicOr()
__device__  int atomicOr(int* address, int val)
{
	return hc::atomic_fetch_or(address,val);
}
__device__  unsigned int atomicOr(unsigned int* address,
                      unsigned int val)
{
	return hc::atomic_fetch_or(address,val);
}
__device__  unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_or((uint64_t*)address,(uint64_t)val);
}

//atomicXor()
__device__  int atomicXor(int* address, int val)
{
	return hc::atomic_fetch_xor(address,val);
}
__device__  unsigned int atomicXor(unsigned int* address,
                       unsigned int val)
{
	return hc::atomic_fetch_xor(address,val);
}
__device__  unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val)
{
	return (long long int)hc::atomic_fetch_xor((uint64_t*)address,(uint64_t)val);
}

//atomicInc
__device__  unsigned int atomicInc(unsigned int* address,
                       unsigned int val)
{
	return hc::__atomic_wrapinc(address,val);
}

//atomicDec
__device__  unsigned int atomicDec(unsigned int* address,
                       unsigned int val)
{
	return hc::__atomic_wrapdec(address,val);
}

//__mul24 __umul24
__device__  int __mul24(int arg1,
                       int arg2)
{
	return hc::__mul24(arg1, arg2);
}
__device__  unsigned int __umul24(unsigned int arg1,
                       unsigned int arg2)
{
	return hc::__mul24(arg1, arg2);
}

__device__ unsigned int test__popc(unsigned int input)
{
    return hc::__popcount_u32_b32(input);
}

// integer intrinsic function __poc __clz __ffs __brev
__device__ unsigned int __popc( unsigned int input)
{
    return hc::__popcount_u32_b32(input);
}

__device__ unsigned int test__popc(unsigned int input);

__device__ unsigned int __popcll( unsigned long long int input)
{
    return hc::__popcount_u32_b64(input);
}

__device__ unsigned int __clz(unsigned int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 32 : hc::__firstbit_u32_u32( input);
#else
    return hc::__firstbit_u32_u32( input);
#endif
}

__device__ unsigned int __clzll(unsigned long long int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 64 : hc::__firstbit_u32_u64( input);
#else
    return hc::__firstbit_u32_u64( input);
#endif
}

__device__ unsigned int __clz( int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 32 : hc::__firstbit_u32_s32( input);
#else
    return hc::__firstbit_u32_s32( input);
#endif
}

__device__ unsigned int __clzll( long long int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 64 : hc::__firstbit_u32_s64( input);
#else
    return hc::__firstbit_u32_s64( input);
#endif
}

__device__ unsigned int __ffs(unsigned int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_u32( input)+1;
#else
    return hc::__lastbit_u32_u32( input);
#endif
}

__device__ unsigned int __ffsll(unsigned long long int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_u64( input)+1;
#else
    return hc::__lastbit_u32_u64( input);
#endif
}

__device__ unsigned int __ffs( int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_s32( input)+1;
#else
    return hc::__lastbit_u32_s32( input);
#endif
}

__device__ unsigned int __ffsll( long long int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_s64( input)+1;
#else
    return hc::__lastbit_u32_s64( input);
#endif
}

__device__ unsigned int __brev( unsigned int input)
{
    return hc::__bitrev_b32( input);
}

__device__ unsigned long long int __brevll( unsigned long long int input)
{
    return hc::__bitrev_b64( input);
}

// warp vote function __all __any __ballot
__device__ int __all(  int input)
{
    return hc::__all( input);
}


__device__ int __any( int input)
{
#ifdef NVCC_COMPAT
    if( hc::__any( input)!=0) return 1;
    else return 0;
#else
    return hc::__any( input);
#endif
}

__device__ unsigned long long int __ballot( int input)
{
    return hc::__ballot( input);
}

// warp shuffle functions
__device__ int __shfl(int input, int lane, int width)
{
  return hc::__shfl(input,lane,width);
}

__device__  int __shfl_up(int input, unsigned int lane_delta, int width)
{
  return hc::__shfl_up(input,lane_delta,width);
}

__device__  int __shfl_down(int input, unsigned int lane_delta, int width)
{
  return hc::__shfl_down(input,lane_delta,width);
}

__device__  int __shfl_xor(int input, int lane_mask, int width)
{
  return hc::__shfl_xor(input,lane_mask,width);
}

__device__  float __shfl(float input, int lane, int width)
{
  return hc::__shfl(input,lane,width);
}

__device__  float __shfl_up(float input, unsigned int lane_delta, int width)
{
  return hc::__shfl_up(input,lane_delta,width);
}

__device__  float __shfl_down(float input, unsigned int lane_delta, int width)
{
  return hc::__shfl_down(input,lane_delta,width);
}

__device__  float __shfl_xor(float input, int lane_mask, int width)
{
  return hc::__shfl_xor(input,lane_mask,width);
}

__host__ __device__ int min(int arg1, int arg2)
{
  return (int)(hc::precise_math::fmin((float)arg1, (float)arg2));
}
__host__ __device__ int max(int arg1, int arg2)
{
  return (int)(hc::precise_math::fmax((float)arg1, (float)arg2));
}

__device__ __attribute__((address_space(3))) void* __get_dynamicgroupbaseptr()
{
  return hc::get_dynamic_group_segment_base_pointer();
}


// Precise Math Functions
__device__ float __hip_precise_cosf(float x) {
  return hc::precise_math::cosf(x);
}

__device__ float __hip_precise_exp10f(float x) {
  return hc::precise_math::exp10f(x);
}

__device__ float __hip_precise_expf(float x) {
  return hc::precise_math::expf(x);
}

__device__ float __hip_precise_frsqrt_rn(float x) {
  return hc::precise_math::rsqrt(x);
}

__device__ float __hip_precise_fsqrt_rd(float x) {
  return hc::precise_math::sqrt(x);
}

__device__ float __hip_precise_fsqrt_rn(float x) {
  return hc::precise_math::sqrt(x);
}

__device__ float __hip_precise_fsqrt_ru(float x) {
  return hc::precise_math::sqrt(x);
}

__device__ float __hip_precise_fsqrt_rz(float x) {
  return hc::precise_math::sqrt(x);
}

__device__ float __hip_precise_log10f(float x) {
  return hc::precise_math::log10(x);
}

__device__ float __hip_precise_log2f(float x) {
  return hc::precise_math::log2(x);
}

__device__ float __hip_precise_logf(float x) {
  return hc::precise_math::logf(x);
}

__device__ float __hip_precise_powf(float base, float exponent) {
  return hc::precise_math::powf(base, exponent);
}

__device__ void __hip_precise_sincosf(float x, float *s, float *c) {
  hc::precise_math::sincosf(x, s, c);
}

__device__ float __hip_precise_sinf(float x) {
  return hc::precise_math::sinf(x);
}

__device__ float __hip_precise_tanf(float x) {
  return hc::precise_math::tanf(x);
}

// Double Precision Math
__device__ double __hip_precise_dsqrt_rd(double x) {
  return hc::precise_math::sqrt(x);
}

__device__ double __hip_precise_dsqrt_rn(double x) {
  return hc::precise_math::sqrt(x);
}

__device__ double __hip_precise_dsqrt_ru(double x) {
  return hc::precise_math::sqrt(x);
}

__device__ double __hip_precise_dsqrt_rz(double x) {
  return hc::precise_math::sqrt(x);
}

#define LOG_BASE2_E_DIV_2 0.4426950408894701
#define LOG_BASE2_5 2.321928094887362
#define ONE_DIV_LOG_BASE2_E 0.69314718056
#define ONE_DIV_LOG_BASE2_10 0.30102999566

// Fast Math Intrinsics
__device__ float __hip_fast_exp10f(float x) {
  return __hip_fast_exp2f(x*LOG_BASE2_E_DIV_2);
}

__device__ float __hip_fast_expf(float x) {
  return __hip_fast_expf(x*LOG_BASE2_5);
}

__device__ float __hip_fast_frsqrt_rn(float x) {
  return 1 / __hip_fast_fsqrt_rd(x);;
}

__device__ float __hip_fast_fsqrt_rn(float x) {
  return __hip_fast_fsqrt_rd(x);
}

__device__ float __hip_fast_fsqrt_ru(float x) {
  return __hip_fast_fsqrt_rd(x);
}

__device__ float __hip_fast_fsqrt_rz(float x) {
  return __hip_fast_fsqrt_rd(x);
}

__device__ float __hip_fast_log10f(float x) {
  return ONE_DIV_LOG_BASE2_E * __hip_fast_log2f(x);
}

__device__ float __hip_fast_logf(float x) {
  return ONE_DIV_LOG_BASE2_10 * __hip_fast_log2f(x);
}

__device__ float __hip_fast_powf(float base, float exponent) {
  return hc::fast_math::powf(base, exponent);
}

__device__ void __hip_fast_sincosf(float x, float *s, float *c) {
  *s = __hip_fast_sinf(x);
  *c = __hip_fast_cosf(x);
}

__device__ float __hip_fast_tanf(float x) {
  return hc::fast_math::tanf(x);
}

// Double Precision Math
__device__ double __hip_fast_dsqrt_rd(double x) {
  return hc::fast_math::sqrt(x);
}

__device__ double __hip_fast_dsqrt_rn(double x) {
  return hc::fast_math::sqrt(x);
}

__device__ double __hip_fast_dsqrt_ru(double x) {
  return hc::fast_math::sqrt(x);
}

__device__ double __hip_fast_dsqrt_rz(double x) {
  return hc::fast_math::sqrt(x);
}

__HIP_DEVICE__ char1 make_char1(signed char x)
{
    char1 c1;
    c1.x = x;
    return c1;
}

__HIP_DEVICE__ char2 make_char2(signed char x, signed char y)
{
    char2 c2;
    c2.x = x;
    c2.y = y;
    return c2;
}

__HIP_DEVICE__ char3 make_char3(signed char x, signed char y, signed char z)
{
    char3 c3;
    c3.x = x;
    c3.y = y;
    c3.z = z;
    return c3;
}

__HIP_DEVICE__ char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
    char4 c4;
    c4.x = x;
    c4.y = y;
    c4.z = z;
    c4.w = w;
    return c4;
}

__HIP_DEVICE__ short1 make_short1(short x)
{
    short1 s1;
    s1.x = x;
    return s1;
}

__HIP_DEVICE__ short2 make_short2(short x, short y)
{
    short2 s2;
    s2.x = x;
    s2.y = y;
    return s2;
}

__HIP_DEVICE__ short3 make_short3(short x, short y, short z)
{
    short3 s3;
    s3.x = x;
    s3.y = y;
    s3.z = z;
    return s3;
}

__HIP_DEVICE__ short4 make_short4(short x, short y, short z, short w)
{
    short4 s4;
    s4.x = x;
    s4.y = y;
    s4.z = z;
    s4.w = w;
    return s4;
}

__HIP_DEVICE__ int1 make_int1(int x)
{
    int1 i1;
    i1.x = x;
    return i1;
}

__HIP_DEVICE__ int2 make_int2(int x, int y)
{
    int2 i2;
    i2.x = x;
    i2.y = y;
    return i2;
}

__HIP_DEVICE__ int3 make_int3(int x, int y, int z)
{
    int3 i3;
    i3.x = x;
    i3.y = y;
    i3.z = z;
    return i3;
}

__HIP_DEVICE__ int4 make_int4(int x, int y, int z, int w)
{
    int4 i4;
    i4.x = x;
    i4.y = y;
    i4.z = z;
    i4.w = w;
    return i4;
}

__HIP_DEVICE__ long1 make_long1(long x)
{
    long1 l1;
    l1.x = x;
    return l1;
}

__HIP_DEVICE__ long2 make_long2(long x, long y)
{
    long2 l2;
    l2.x = x;
    l2.y = y;
    return l2;
}

__HIP_DEVICE__ long3 make_long3(long x, long y, long z)
{
    long3 l3;
    l3.x = x;
    l3.y = y;
    l3.z = z;
    return l3;
}

__HIP_DEVICE__ long4 make_long4(long x, long y, long z, long w)
{
    long4 l4;
    l4.x = x;
    l4.y = y;
    l4.z = z;
    l4.w = w;
    return l4;
}

__HIP_DEVICE__ longlong1 make_longlong1(long long x)
{
    longlong1 l1;
    l1.x = x;
    return l1;
}

__HIP_DEVICE__ longlong2 make_longlong2(long long x, long long y)
{
    longlong2 l2;
    l2.x = x;
    l2.y = y;
    return l2;
}

__HIP_DEVICE__ longlong3 make_longlong3(long long x, long long y, long long z)
{
    longlong3 l3;
    l3.x = x;
    l3.y = y;
    l3.z = z;
    return l3;
}

__HIP_DEVICE__ longlong4 make_longlong4(long long x, long long y, long long z, long long w)
{
    longlong4 l4;
    l4.x = x;
    l4.y = y;
    l4.z = z;
    l4.w = w;
    return l4;
}

__HIP_DEVICE__ uchar1 make_uchar1(unsigned char x)
{
    uchar1 c1;
    c1.x = x;
    return c1;
}

__HIP_DEVICE__ uchar2 make_uchar2(unsigned char x, unsigned char y)
{
    uchar2 c2;
    c2.x = x;
    c2.y = y;
    return c2;
}

__HIP_DEVICE__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
    uchar3 c3;
    c3.x = x;
    c3.y = y;
    c3.z = z;
    return c3;
}

__HIP_DEVICE__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
    uchar4 c4;
    c4.x = x;
    c4.y = y;
    c4.z = z;
    c4.w = w;
    return c4;
}

__HIP_DEVICE__ ushort1 make_ushort1(unsigned short x)
{
    ushort1 s1;
    s1.x = x;
    return s1;
}

__HIP_DEVICE__ ushort2 make_ushort2(unsigned short x, unsigned short y)
{
    ushort2 s2;
    s2.x = x;
    s2.y = y;
    return s2;
}

__HIP_DEVICE__ ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
    ushort3 s3;
    s3.x = x;
    s3.y = y;
    s3.z = z;
    return s3;
}

__HIP_DEVICE__ ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
    ushort4 s4;
    s4.x = x;
    s4.y = y;
    s4.z = z;
    s4.w = w;
    return s4;
}

__HIP_DEVICE__ uint1 make_uint1(unsigned int x)
{
    uint1 i1;
    i1.x = x;
    return i1;
}

__HIP_DEVICE__ uint2 make_uint2(unsigned int x, unsigned int y)
{
    uint2 i2;
    i2.x = x;
    i2.y = y;
    return i2;
}

__HIP_DEVICE__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
    uint3 i3;
    i3.x = x;
    i3.y = y;
    i3.z = z;
    return i3;
}

__HIP_DEVICE__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
    uint4 i4;
    i4.x = x;
    i4.y = y;
    i4.z = z;
    i4.w = w;
    return i4;
}

__HIP_DEVICE__ ulong1 make_ulong1(unsigned long x)
{
    ulong1 l1;
    l1.x = x;
    return l1;
}

__HIP_DEVICE__ ulong2 make_ulong2(unsigned long x, unsigned long y)
{
    ulong2 l2;
    l2.x = x;
    l2.y = y;
    return l2;
}

__HIP_DEVICE__ ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z)
{
    ulong3 l3;
    l3.x = x;
    l3.y = y;
    l3.z = z;
    return l3;
}

__HIP_DEVICE__ ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w)
{
    ulong4 l4;
    l4.x = x;
    l4.y = y;
    l4.z = z;
    l4.w = w;
    return l4;
}

__HIP_DEVICE__ ulonglong1 make_ulonglong1(unsigned long long x)
{
    ulonglong1 l1;
    l1.x = x;
    return l1;
}

__HIP_DEVICE__ ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y)
{
    ulonglong2 l2;
    l2.x = x;
    l2.y = y;
    return l2;
}

__HIP_DEVICE__ ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z)
{
    ulonglong3 l3;
    l3.x = x;
    l3.y = y;
    l3.z = z;
    return l3;
}

__HIP_DEVICE__ ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w)
{
    ulonglong4 l4;
    l4.x = x;
    l4.y = y;
    l4.z = z;
    l4.w = w;
    return l4;
}

__HIP_DEVICE__ float1 make_float1(float x)
{
    float1 f1;
    f1.x = x;
    return f1;
}

__HIP_DEVICE__ float2 make_float2(float x, float y)
{
    float2 f2;
    f2.x = x;
    f2.y = y;
    return f2;
}

__HIP_DEVICE__ float3 make_float3(float x, float y, float z)
{
    float3 f3;
    f3.x = x;
    f3.y = y;
    f3.z = z;
    return f3;
}

__HIP_DEVICE__ float4 make_float4(float x, float y, float z, float w)
{
    float4 f4;
    f4.x = x;
    f4.y = y;
    f4.z = z;
    f4.w = w;
    return f4;
}

__HIP_DEVICE__ double1 make_double1(double x)
{
    double1 d1;
    d1.x = x;
    return d1;
}

__HIP_DEVICE__ double2 make_double2(double x, double y)
{
    double2 d2;
    d2.x = x;
    d2.y = y;
    return d2;
}

__HIP_DEVICE__ double3 make_double3(double x, double y, double z)
{
    double3 d3;
    d3.x = x;
    d3.y = y;
    d3.z = z;
    return d3;
}

__HIP_DEVICE__ double4 make_double4(double x, double y, double z, double w)
{
    double4 d4;
    d4.x = x;
    d4.y = y;
    d4.z = z;
    d4.w = w;
    return d4;
}

__device__ void  __threadfence_system(void){
    // no-op
}

float __hip_host_erfinvf(float x)
{
    float ret;
    int  sign;
    if (x < -1 || x > 1){
        return NAN;
    }
    if (x == 0){
        return 0;
    }
    if (x > 0){
        sign = 1;
    } else {
        sign = -1;
        x = -x;
    }
    if (x <= 0.7) {
        float x1 = x * x;
        float x2 = std::fma(__hip_erfinva3, x1, __hip_erfinva2);
        float x3 = std::fma(x2, x1, __hip_erfinva1);
        float x4 = x * std::fma(x3, x1, __hip_erfinva0);

        float r1 = std::fma(__hip_erfinvb4, x1, __hip_erfinvb3);
        float r2 = std::fma(r1, x1, __hip_erfinvb2);
        float r3 = std::fma(r2, x1, __hip_erfinvb1);
        ret = x4 / std::fma(r3, x1, __hip_erfinvb0);
    } else {
        float x1 = std::sqrt(-std::log((1 - x) / 2));
        float x2 = std::fma(__hip_erfinvc3, x1, __hip_erfinvc2);
        float x3 = std::fma(x2, x1, __hip_erfinvc1);
        float x4 = std::fma(x3, x1, __hip_erfinvc0);

        float r1 = std::fma(__hip_erfinvd2, x1, __hip_erfinvd1);
        ret = x4 / std::fma(r1, x1, __hip_erfinvd0);
    }

    ret = ret * sign;
    x = x * sign;

    ret -= (std::erf(ret) - x) / (2 / std::sqrt(HIP_PI) * std::exp(-ret * ret));
    ret -= (std::erf(ret) - x) / (2 / std::sqrt(HIP_PI) * std::exp(-ret * ret));

    return ret;

}

double __hip_host_erfinv(double x)
{
    double ret;
    int  sign;
    if (x < -1 || x > 1){
        return NAN;
    }
    if (x == 0){
        return 0;
    }
    if (x > 0){
        sign = 1;
    } else {
        sign = -1;
        x = -x;
    }
    if (x <= 0.7) {
        double x1 = x * x;
        double x2 = std::fma(__hip_erfinva3, x1, __hip_erfinva2);
        double x3 = std::fma(x2, x1, __hip_erfinva1);
        double x4 = x * std::fma(x3, x1, __hip_erfinva0);

        double r1 = std::fma(__hip_erfinvb4, x1, __hip_erfinvb3);
        double r2 = std::fma(r1, x1, __hip_erfinvb2);
        double r3 = std::fma(r2, x1, __hip_erfinvb1);
        ret = x4 / std::fma(r3, x1, __hip_erfinvb0);
    } else {
        double x1 = std::sqrt(-std::log((1 - x) / 2));
        double x2 = std::fma(__hip_erfinvc3, x1, __hip_erfinvc2);
        double x3 = std::fma(x2, x1, __hip_erfinvc1);
        double x4 = std::fma(x3, x1, __hip_erfinvc0);

        double r1 = std::fma(__hip_erfinvd2, x1, __hip_erfinvd1);
        ret = x4 / std::fma(r1, x1, __hip_erfinvd0);
    }

    ret = ret * sign;
    x = x * sign;

    ret -= (std::erf(ret) - x) / (2 / std::sqrt(HIP_PI) * std::exp(-ret * ret));
    ret -= (std::erf(ret) - x) / (2 / std::sqrt(HIP_PI) * std::exp(-ret * ret));

    return ret;

}

float __hip_host_erfcinvf(float y)
{
    return __hip_host_erfinvf(1 - y);
}

double __hip_host_erfcinv(double y)
{
    return __hip_host_erfinv(1 - y);
}

double __hip_host_j0(double x)
{
    double ret, a = std::fabs(x);
    if (a < 8.0) {
        double y = x*x;
        double y1 = __hip_j0a6 * y + __hip_j0a5;
        double z1 = 1.0 * y + __hip_j0b5;

        double y2 = y1 * y + __hip_j0a4;
        double z2 = z1 * y + __hip_j0b4;

        double y3 = y2 * y + __hip_j0a3;
        double z3 = z2 * y + __hip_j0b3;

        double y4 = y3 * y + __hip_j0a2;
        double z4 = z3 * y + __hip_j0b2;

        double y5 = y4 * y + __hip_j0a1;
        double z5 = z4 * y + __hip_j0b1;

        ret = y5 / z5;

    }
    else {
        double z = 8.0 / a;
        double y = z*z;
        double x1 = a - __hip_j0c;

        double y1 = __hip_j0c4 * y + __hip_j0c3;
        double z1 = __hip_j0d5 * y + __hip_j0d4;

        double y2 = y1 * y + __hip_j0c2;
        double z2 = z1 * z + __hip_j0d3;

        double y3 = y2 * y + __hip_j0c1;
        double z3 = z2 * y + __hip_j0d2;

        double y4 = y3 * y + 1.0;
        double z4 = z3 * y + __hip_j0d1;

        ret = std::sqrt(__hip_j0e / a)*(std::cos(x1) * y4 - z * std::sin(x1) * z4);
    }
    return ret;
}

float __hip_host_j0f(float x)
{
    float ret, a = fabs(x);
    if (a < 8.0) {
        float y = x*x;
        float y1 = __hip_j0a6 * y + __hip_j0a5;
        float z1 = 1.0 * y + __hip_j0b5;

        float y2 = y1 * y + __hip_j0a4;
        float z2 = z1 * y + __hip_j0b4;

        float y3 = y2 * y + __hip_j0a3;
        float z3 = z2 * y + __hip_j0b3;

        float y4 = y3 * y + __hip_j0a2;
        float z4 = z3 * y + __hip_j0b2;

        float y5 = y4 * y + __hip_j0a1;
        float z5 = z4 * y + __hip_j0b1;

        ret = y5 / z5;

    }
    else {
        float z = 8.0 / a;
        float y = z*z;
        float x1 = a - __hip_j0c;

        float y1 = __hip_j0c4 * y + __hip_j0c3;
        float z1 = __hip_j0d5 * y + __hip_j0d4;

        float y2 = y1 * y + __hip_j0c2;
        float z2 = z1 * z + __hip_j0d3;

        float y3 = y2 * y + __hip_j0c1;
        float z3 = z2 * y + __hip_j0d2;

        float y4 = y3 * y + 1.0;
        float z4 = z3 * y + __hip_j0d1;

        ret = std::sqrt(__hip_j0e / a)*(std::cos(x1) * y4 - z * std::sin(x1) * z4);
    }
    return ret;
}

double __hip_host_j1(double x)
{
    double ret, a = std::fabs(x);
    if (a < 8.0) {
        double y = x*x;

        double y1 = __hip_j1a1 * y + __hip_j1a2;
        double z1 = 1.0 * y + __hip_j1b1;

        double y2 = y1 * y + __hip_j1a3;
        double z2 = z1 * y + __hip_j1b2;

        double y3 = y2 * y + __hip_j1a4;
        double z3 = z2 * y + __hip_j1b3;

        double y4 = y3 * y + __hip_j1a5;
        double z4 = z3 * y + __hip_j1b4;

        double y5 = y4 * y + __hip_j1a6;
        double z5 = z4 * y + __hip_j1b5;

        ret = x * y5 / z5;

    }
    else {
        double z = 8.0 / a;
        double y = z*z;
        double x1 = a - __hip_j1c;

        double y1 = __hip_j1c1 * y + __hip_j1c2;
        double y2 = y1 * y + __hip_j1c3;
        double y3 = y2 * y + __hip_j1c4;
        double y4 = y3 * y + 1.0;

        double z1 = __hip_j1d1 * y + __hip_j1d2;
        double z2 = z1 * y + __hip_j1d3;
        double z3 = z2 * y + __hip_j1d4;
        double z4 = z3 * y + __hip_j1d5;

        ret = std::sqrt(__hip_j1e / a)*(std::cos(x1)*y4 - z*std::sin(x1)*z4);
        if (x < 0.0) ret = -ret;
    }
    return ret;
}

float __hip_host_j1f(float x)
{
    double ret, a = fabs(x);
    if (a < 8.0) {
        float y = x*x;

        float y1 = __hip_j1a1 * y + __hip_j1a2;
        float z1 = 1.0 * y + __hip_j1b1;

        float y2 = y1 * y + __hip_j1a3;
        float z2 = z1 * y + __hip_j1b2;

        float y3 = y2 * y + __hip_j1a4;
        float z3 = z2 * y + __hip_j1b3;

        float y4 = y3 * y + __hip_j1a5;
        float z4 = z3 * y + __hip_j1b4;

        float y5 = y4 * y + __hip_j1a6;
        float z5 = z4 * y + __hip_j1b5;

        ret = x * y5 / z5;

    }
    else {
        float z = 8.0 / a;
        float y = z*z;
        float x1 = a - __hip_j1c;

        float y1 = __hip_j1c1 * y + __hip_j1c2;
        float y2 = y1 * y + __hip_j1c3;
        float y3 = y2 * y + __hip_j1c4;
        float y4 = y3 * y + 1.0;

        float z1 = __hip_j1d1 * y + __hip_j1d2;
        float z2 = z1 * y + __hip_j1d3;
        float z3 = z2 * y + __hip_j1d4;
        float z4 = z3 * y + __hip_j1d5;

        ret = std::sqrt(__hip_j1e / a)*(std::cos(x1)*y4 - z*std::sin(x1)*z4);
        if (x < 0.0) ret = -ret;
    }
    return ret;
}

double __hip_host_y0(double x)
{
    double ret;

    if (x < 8.0) {
        double y = x*x;
        double y1 = __hip_y0a1 * y + __hip_y0a2;
        double y2 = y1 * y + __hip_y0a3;
        double y3 = y2 * y + __hip_y0a4;
        double y4 = y3 * y + __hip_y0a5;
        double y5 = y4 * y + __hip_y0a6;

        double z1 = 1.0 * y + __hip_y0b1;
        double z2 = z1 * y + __hip_y0b2;
        double z3 = z2 * y + __hip_y0b3;
        double z4 = z3 * y + __hip_y0b4;
        double z5 = z4 * y + __hip_y0b5;


        ret = (y5 / z5) + __hip_y0c * __hip_host_j0(x) * std::log(x);
    }
    else {
        double z = 8.0 / x;
        double y = z*z;
        double x1 = x - __hip_y0d;

        double y1 = __hip_y0e1 * y + __hip_y0e2;
        double y2 = y1 * y + __hip_y0e3;
        double y3 = y2 * y + __hip_y0e4;
        double y4 = y3 * y + 1.0;

        double z1 = __hip_y0f1 * y + __hip_y0f2;
        double z2 = z1 * y + __hip_y0f3;
        double z3 = z2 * y + __hip_y0f4;
        double z4 = z3 * y + __hip_y0f5;

        ret = std::sqrt(__hip_y1g / x)*(std::sin(x1)*y4 + z * std::cos(x1) * z4);
    }
    return ret;

}

float __hip_host_y0f(float x)
{
    float ret;

    if (x < 8.0) {
        float y = x*x;
        float y1 = __hip_y0a1 * y + __hip_y0a2;
        float y2 = y1 * y + __hip_y0a3;
        float y3 = y2 * y + __hip_y0a4;
        float y4 = y3 * y + __hip_y0a5;
        float y5 = y4 * y + __hip_y0a6;

        float z1 = 1.0 * y + __hip_y0b1;
        float z2 = z1 * y + __hip_y0b2;
        float z3 = z2 * y + __hip_y0b3;
        float z4 = z3 * y + __hip_y0b4;
        float z5 = z4 * y + __hip_y0b5;


        ret = (y5 / z5) + __hip_y0c * __hip_host_j0f(x) * log(x);
    }
    else {
        float z = 8.0 / x;
        float y = z*z;
        float x1 = x - __hip_y0d;

        float y1 = __hip_y0e1 * y + __hip_y0e2;
        float y2 = y1 * y + __hip_y0e3;
        float y3 = y2 * y + __hip_y0e4;
        float y4 = y3 * y + 1.0;

        float z1 = __hip_y0f1 * y + __hip_y0f2;
        float z2 = z1 * y + __hip_y0f3;
        float z3 = z2 * y + __hip_y0f4;
        float z4 = z3 * y + __hip_y0f5;

        ret = std::sqrt(__hip_y1g / x)*(std::sin(x1)*y4 + z * std::cos(x1) * z4);
    }
    return ret;

}

double __hip_host_y1(double x)
{
    double ret;

    if (x < 8.0) {
        double y = x*x;

        double y1 = __hip_y1a1 * y + __hip_y1a2;
        double y2 = y1 * y + __hip_y1a3;
        double y3 = y2 * y + __hip_y1a4;
        double y4 = y3 * y + __hip_y1a5;
        double y5 = y4 * y + __hip_y1a6;
        double y6 = x * y5;

        double z1 = __hip_y1b1 + y;
        double z2 = z1 * y + __hip_y1b2;
        double z3 = z2 * y + __hip_y1b3;
        double z4 = z3 * y + __hip_y1b4;
        double z5 = z4 * y + __hip_y1b5;
        double z6 = z5 * y + __hip_y1b6;

        ret = (y6 / z6) + __hip_y1c * (__hip_host_j1(x) * std::log(x) - 1.0 / x);
    }
    else {
        double z = 8.0 / x;
        double y = z*z;
        double x1 = x - __hip_y1d;

        double y1 = __hip_y1e1 * y + __hip_y1e2;
        double y2 = y1 * y + __hip_y1e3;
        double y3 = y2 * y + __hip_y1e4;
        double y4 = y3 * y + 1.0;

        double z1 = __hip_y1f1 * y + __hip_y1f2;
        double z2 = z1 * y + __hip_y1f3;
        double z3 = z2 * y + __hip_y1f4;
        double z4 = z3 * y + __hip_y1f5;

        ret = std::sqrt(__hip_y1g / x)*(std::sin(x1)*y4 + z * std::cos(x1)*z4);
    }
    return ret;
}

float __hip_host_y1f(float x)
{
    float ret;

    if (x < 8.0) {
        float y = x*x;

        float y1 = __hip_y1a1 * y + __hip_y1a2;
        float y2 = y1 * y + __hip_y1a3;
        float y3 = y2 * y + __hip_y1a4;
        float y4 = y3 * y + __hip_y1a5;
        float y5 = y4 * y + __hip_y1a6;
        float y6 = x * y5;

        float z1 = __hip_y1b1 + y;
        float z2 = z1 * y + __hip_y1b2;
        float z3 = z2 * y + __hip_y1b3;
        float z4 = z3 * y + __hip_y1b4;
        float z5 = z4 * y + __hip_y1b5;
        float z6 = z5 * y + __hip_y1b6;

        ret = (y6 / z6) + __hip_y1c * (__hip_host_j1f(x) * log(x) - 1.0 / x);
    }
    else {
        float z = 8.0 / x;
        float y = z*z;
        float x1 = x - __hip_y1d;

        float y1 = __hip_y1e1 * y + __hip_y1e2;
        float y2 = y1 * y + __hip_y1e3;
        float y3 = y2 * y + __hip_y1e4;
        float y4 = y3 * y + 1.0;

        float z1 = __hip_y1f1 * y + __hip_y1f2;
        float z2 = z1 * y + __hip_y1f3;
        float z3 = z2 * y + __hip_y1f4;
        float z4 = z3 * y + __hip_y1f5;

        ret = std::sqrt(__hip_y1g / x)*(std::sin(x1)*y4 + z*std::cos(x1)*z4);
    }
    return ret;
}

double __hip_host_jn(int n, double x)
{
    int    sum = 0, m;
    double a, b0, b1, b2, val, t, ret;
    if (n < 0){
        return NAN;
    }
    a = std::fabs(x);
    if (n == 0){
        return(__hip_host_j0(a));
    }
    if (n == 1){
        return(__hip_host_j1(a));
    }
    if (a == 0.0){
        return 0.0;
    }
    else if (a > (double)n) {
        t = 2.0 / a;
        b1 = __hip_host_j0(a);
        b0 = __hip_host_j1(a);
        for (int i = 1; i<n; i++) {
            b2 = i*t*b0 - b1;
            b1 = b0;
            b0 = b2;
        }
        ret = b0;
    }
    else {
        t = 2.0 / a;
        m = 2 * ((n + (int)std::sqrt(__hip_k1*n)) / 2);
        b2 = ret = val = 0.0;
        b0 = 1.0;
        for (int i = m; i>0; i--) {
            b1 = i*t*b0 - b2;
            b2 = b0;
            b0 = b1;
            if (std::fabs(b0) > __hip_k2) {
                b0 *= __hip_k3;
                b2 *= __hip_k3;
                ret *= __hip_k3;
                val *= __hip_k3;
            }
            if (sum) val += b0;
            sum = !sum;
            if (i == n) ret = b2;
        }
        val = 2.0*val - b0;
        ret /= val;
    }
    return  x < 0.0 && n % 2 == 1 ? -ret : ret;
}

float __hip_host_jnf(int n, float x)
{
    int    sum = 0, m;
    float a, b0, b1, b2, val, t, ret;
    if (n < 0){
        return NAN;
    }
    a = fabs(x);
    if (n == 0){
        return(__hip_host_j0f(a));
    }
    if (n == 1){
        return(__hip_host_j1f(a));
    }
    if (a == 0.0){
        return 0.0;
    }
    else if (a > (float)n) {
        t = 2.0 / a;
        b1 = __hip_host_j0f(a);
        b0 = __hip_host_j1f(a);
        for (int i = 1; i<n; i++) {
            b2 = i*t*b0 - b1;
            b1 = b0;
            b0 = b2;
        }
        ret = b0;
    }
    else {
        t = 2.0 / a;
        m = 2 * ((n + (int)std::sqrt(__hip_k1*n)) / 2);
        b2 = ret = val = 0.0;
        b0 = 1.0;
        for (int i = m; i>0; i--) {
            b1 = i*t*b0 - b2;
            b2 = b0;
            b0 = b1;
            if (std::fabs(b0) > __hip_k2) {
                b0 *= __hip_k3;
                b2 *= __hip_k3;
                ret *= __hip_k3;
                val *= __hip_k3;
            }
            if (sum) val += b0;
            sum = !sum;
            if (i == n) ret = b2;
        }
        val = 2.0*val - b0;
        ret /= val;
    }
    return  x < 0.0 && n % 2 == 1 ? -ret : ret;
}

double __hip_host_yn(int n, double x)
{
    double b0, b1, b2, t;

    if (n < 0 || x == 0.0)
    {
        return NAN;
    }
    if (n == 0){
        return(__hip_host_y0(x));
    }
    if (n == 1){
        return(__hip_host_y1(x));
    }
    t = 2.0 / x;
    b0 = __hip_host_y1(x);
    b1 = __hip_host_y0(x);
    for (int i = 1; i<n; i++) {
        b2 = i*t*b0 - b1;
        b1 = b0;
        b0 = b2;
    }
    return b0;
}

float __hip_host_ynf(int n, float x)
{
    float b0, b1, b2, t;

    if (n < 0 || x == 0.0)
    {
        return NAN;
    }
    if (n == 0){
        return(__hip_host_y0f(x));
    }
    if (n == 1){
        return(__hip_host_y1f(x));
    }
    t = 2.0 / x;
    b0 = __hip_host_y1f(x);
    b1 = __hip_host_y0f(x);
    for (int i = 1; i<n; i++) {
        b2 = i*t*b0 - b1;
        b1 = b0;
        b0 = b2;
    }
    return b0;
}

__host__ float modff(float x, float *iptr)
{
    return std::modf(x, iptr);
}

__host__ float erfcinvf(float y)
{
    return __hip_host_erfcinvf(y);
}

__host__ double erfcinv(double y)
{
    return __hip_host_erfcinv(y);
}

__host__ float erfinvf(float x)
{
    return __hip_host_erfinvf(x);
}

__host__ double erfinv(double x)
{
    return __hip_host_erfinv(x);
}

__host__ double fdivide(double x, double y)
{
    return x/y;
}

__host__ float normcdff(float t)
{
     return (1 - std::erf(-t/std::sqrt(2)))/2;
}

__host__ double normcdf(double x)
{
     return (1 - std::erf(-x/std::sqrt(2)))/2;
}

__host__ float erfcxf(float x)
{
     return std::exp(x*x) * std::erfc(x);
}

__host__ double erfcx(double x)
{
     return std::exp(x*x) * std::erfc(x);
}

__host__ float rhypotf(float x, float y)
{
     return 1 / std::sqrt(x*x + y*y);
}

__host__ double rhypot(double x, double y)
{
    return 1 / std::sqrt(x*x + y*y);
}

__host__ float rcbrtf(float a)
{
    return 1 / std::cbrt(a);
}

__host__ double rcbrt(double a)
{
    return 1 / std::cbrt(a);
}

__host__ float normf(int dim, const float *a)
{
    float val = 0.0f;
    for(int i=0;i<dim;i++)
    {
        val = val + a[i] * a[i];
    }
    return val;
}

__host__ double norm(int dim, const double *a)
{
    double val = 0.0;
    for(int i=0;i<dim;i++)
    {
        val = val + a[i] * a[i];
    }
    return val;
}

__host__ float rnormf(int dim, const float *t)
{
    float val = 0.0f;
    for(int i=0;i<dim;i++)
    {
        val = val + t[i] * t[i];
    }
    return 1 / std::sqrt(val);
}

__host__ double rnorm(int dim, const double *t)
{
    double val = 0.0;
    for(int i=0;i<dim;i++)
    {
        val = val + t[i] * t[i];
    }
    return 1 / std::sqrt(val);
}

__host__ float rnorm4df(float a, float b, float c, float d)
{
    return 1 / std::sqrt(a*a + b*b + c*c + d*d);
}

__host__ double rnorm4d(double a, double b, double c, double d)
{
    return 1 / std::sqrt(a*a + b*b + c*c + d*d);
}

__host__ float rnorm3df(float a, float b, float c)
{
    return 1 / std::sqrt(a*a + b*b + c*c);
}

__host__ double rnorm3d(double a, double b, double c)
{
    return 1 / std::sqrt(a*a + b*b + c*c);
}

__host__ void sincospif(float x, float *sptr, float *cptr)
{
    *sptr = std::sin(HIP_PI*x);
    *cptr = std::cos(HIP_PI*x);
}

__host__ void sincospi(double x, double *sptr, double *cptr)
{
    *sptr = std::sin(HIP_PI*x);
    *cptr = std::cos(HIP_PI*x);
}

__host__ float normcdfinvf(float x)
{
    return std::sqrt(2) * erfinv(2*x-1);
}

__host__ double normcdfinv(double x)
{
    return std::sqrt(2) * erfinv(2*x-1);
}

__host__ float nextafterf(float x, float y)
{
    return std::nextafter(x, y);
}

__host__ double nextafter(double x, double y)
{
    return std::nextafter(x, y);
}

__host__ float norm3df(float a, float b, float c)
{
    return std::sqrt(a*a + b*b + c*c);
}

__host__ float norm4df(float a, float b, float c, float d)
{
    return std::sqrt(a*a + b*b + c*c + d*d);
}

__host__ double norm3d(double a, double b, double c)
{
    return std::sqrt(a*a + b*b + c*c);
}

__host__ double norm4d(double a, double b, double c, double d)
{
    return std::sqrt(a*a + b*b + c*c + d*d);
}
