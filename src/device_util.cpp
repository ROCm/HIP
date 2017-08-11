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

#include <hc.hpp>
#include <grid_launch.h>
#include <hc_math.hpp>
#include "device_util.h"
#include "hip/hcc_detail/device_functions.h"
#include "hip/hip_runtime.h"
#include <atomic>

//=================================================================================================
/*
 Implementation of malloc and free device functions.

 This is the best place to put them because the device
 global variables need to be initialized at the start.
*/
__device__ char gpuHeap[SIZE_OF_HEAP];
__device__ uint32_t gpuFlags[NUM_PAGES];

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



// loop unrolling
__device__ void* __hip_hc_memcpy(void* dst, const void* src, size_t size)
{
    uint8_t *dstPtr, *srcPtr;
    dstPtr = (uint8_t*)dst;
    srcPtr = (uint8_t*)src;
    for(uint32_t i=0;i<size;i++) {
        dstPtr[i] = srcPtr[i];
    }
    return nullptr;
}

__device__ void* __hip_hc_memset(void* ptr, uint8_t val, size_t size)
{
    uint8_t *dstPtr;
    dstPtr = (uint8_t*)ptr;
    for(uint32_t i=0;i<size;i++) {
        dstPtr[i] = val;
    }
    return nullptr;
}

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

__device__ long long int clock64() { return (long long int)hc::__cycle_u64(); };
__device__ clock_t clock() { return (clock_t)hc::__cycle_u64(); };

//abort
__device__ void abort()
{
    return hc::abort();
}

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
template<typename T>
__device__ T atomicCAS_impl(T* address, T compare, T val)
{
  // the implementation assumes the atomic is lock-free and 
  // has the same size as the non-atmoic equivalent type
  static_assert(sizeof(T) == sizeof(std::atomic<T>)
                , "size mismatch between atomic and non-atomic types");

  union {
    T*              address;
    std::atomic<T>* atomic_address;
  } u;
  u.address = address;

  T expected = compare;

  // hcc should generate a system scope atomic CAS 
  std::atomic_compare_exchange_weak_explicit(u.atomic_address
                                            , &expected, val
                                            , std::memory_order_acq_rel
                                            , std::memory_order_relaxed);
  return expected;
}

__device__  int atomicCAS(int* address, int compare, int val)
{
  return atomicCAS_impl(address, compare, val);
}
__device__  unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val)
{
  return atomicCAS_impl(address, compare, val);
}
__device__  unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val)
{
  return atomicCAS_impl(address, compare, val);
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

__device__ void* __get_dynamicgroupbaseptr() {
  return hc::get_dynamic_group_segment_base_pointer();
}

__host__ void* __get_dynamicgroupbaseptr() { 
  return nullptr; 
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

#define LOG_BASE2_E 1.4426950408889634
#define LOG_BASE2_10 3.32192809488736
#define ONE_DIV_LOG_BASE2_E 0.69314718056
#define ONE_DIV_LOG_BASE2_10 0.30102999566

// Fast Math Intrinsics
__device__ float __hip_fast_exp10f(float x) {
  return __hip_fast_exp2f(x*LOG_BASE2_E);
}

__device__ float __hip_fast_expf(float x) {
  return __hip_fast_exp2f(x*LOG_BASE2_10);
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
// FIXME - HCC doesn't have a fast_math version double FP sqrt
// Another issue is that these intrinsics call for a specific rounding mode;
// however, their implementation all map to the same sqrt builtin
__device__ double __hip_fast_dsqrt_rd(double x) {
  return hc::precise_math::sqrt(x);
}

__device__ double __hip_fast_dsqrt_rn(double x) {
  return hc::precise_math::sqrt(x);
}

__device__ double __hip_fast_dsqrt_ru(double x) {
  return hc::precise_math::sqrt(x);
}

__device__ double __hip_fast_dsqrt_rz(double x) {
  return hc::precise_math::sqrt(x);
}

__device__ void  __threadfence_system(void){
  std::atomic_thread_fence(std::memory_order_seq_cst);
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
