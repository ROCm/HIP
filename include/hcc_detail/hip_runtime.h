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
/**
 *  @file  hcc_detail/hip_runtime.h
 *  @brief Contains definitions of APIs for HIP runtime.
 */

#pragma once

//---
// Top part of file can be compiled with any compiler


//#include <cstring>
//#include <cmath>
#include <string.h>
#include <stddef.h>


#define CUDA_SUCCESS hipSuccess

#include <hip_runtime_api.h>
//#include "hcc_detail/hip_hcc.h"
//---
// Remainder of this file only compiles with HCC
#ifdef __HCC__
//#if __cplusplus
#include <hc.hpp>
//#endif
#include <grid_launch.h>
extern int HIP_TRACE_API;

//TODO-HCC-GL - change this to typedef.
//typedef grid_launch_parm hipLaunchParm ;
#define hipLaunchParm grid_launch_parm

#include <hcc_detail/hip_texture.h>
#include <hcc_detail/host_defines.h>
// TODO-HCC remove old definitions ; ~1602 hcc supports __HCC_ACCELERATOR__ define.
#if defined (__KALMAR_ACCELERATOR__) && !defined (__HCC_ACCELERATOR__)
#define __HCC_ACCELERATOR__  __KALMAR_ACCELERATOR__
#endif

// Feature tests:
#if defined(__HCC_ACCELERATOR__) && (__HCC_ACCELERATOR__ != 0)
// Device compile and not host compile:

//TODO-HCC enable __HIP_ARCH_HAS_ATOMICS__ when HCC supports these.
    // 32-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__       (1)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__   (1)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__       (1)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__   (1)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__           (0)

// 64-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__       (1)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__       (0)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__                    (1)

//warp cross-lane operations:
#define __HIP_ARCH_HAS_WARP_VOTE__                  (1)
#define __HIP_ARCH_HAS_WARP_BALLOT__                (1)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__               (1)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__          (0)

//sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__        (0)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__            (0)

// misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__              (0)
#define __HIP_ARCH_HAS_3DGRID__                     (1)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__           (0)

#endif





//TODO-HCC  this is currently ignored by HCC target of HIP
#define __launch_bounds__(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)

// Detect if we are compiling C++ mode or C mode
#if defined(__cplusplus)
#define __HCC_CPP__
#elif defined(__STDC_VERSION__)
#define __HCC_C__
#endif


// TODO - hipify-clang - change to use the function call.
//#define warpSize hc::__wavesize()
extern const int warpSize;


#define clock_t long long int
__device__ long long int clock64();
__device__ clock_t clock();

//atomicAdd()
__device__ int atomicAdd(int* address, int val);
__device__ unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);

__device__ unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);

__device__ float atomicAdd(float* address, float val);


//atomicSub()
__device__ int atomicSub(int* address, int val);

__device__ unsigned int atomicSub(unsigned int* address,
                       unsigned int val);


//atomicExch()
__device__ int atomicExch(int* address, int val);

__device__ unsigned int atomicExch(unsigned int* address,
                        unsigned int val);

__device__ unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);

__device__ float atomicExch(float* address, float val);


//atomicMin()
__device__ int atomicMin(int* address, int val);
__device__ unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);


//atomicMax()
__device__ int atomicMax(int* address, int val);
__device__ unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);


//atomicCAS()
__device__ int atomicCAS(int* address, int compare, int val);
__device__ unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
__device__ unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);


//atomicAnd()
__device__ int atomicAnd(int* address, int val);
__device__ unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);


//atomicOr()
__device__ int atomicOr(int* address, int val);
__device__ unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
__device__ unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);


//atomicXor()
__device__ int atomicXor(int* address, int val);
__device__ unsigned int atomicXor(unsigned int* address,
                       unsigned int val);
__device__ unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val);


#include <hc.hpp>
// integer intrinsic function __poc __clz __ffs __brev
__device__ unsigned int __popc( unsigned int input);
__device__ unsigned int __popcll( unsigned long long int input);
__device__ unsigned int __clz(unsigned int input);
__device__ unsigned int __clzll(unsigned long long int input);
__device__ unsigned int __clz(int input);
__device__ unsigned int __clzll(long long int input);
__device__ unsigned int __ffs(unsigned int input);
__device__ unsigned int __ffsll(unsigned long long int input);
__device__ unsigned int __ffs(int input);
__device__ unsigned int __ffsll(long long int input);
__device__ unsigned int __brev( unsigned int input);
__device__ unsigned long long int __brevll( unsigned long long int input);


// warp vote function __all __any __ballot
__device__ int __all(  int input);
__device__ int __any( int input);
__device__  unsigned long long int __ballot( int input);

// warp shuffle functions
#ifdef __cplusplus

__device__ int __shfl(int input, int lane, int width=warpSize);
__device__ int __shfl_up(int input, unsigned int lane_delta, int width=warpSize);
__device__ int __shfl_down(int input, unsigned int lane_delta, int width=warpSize);
__device__ int __shfl_xor(int input, int lane_mask, int width=warpSize);
__device__ float __shfl(float input, int lane, int width=warpSize);
__device__ float __shfl_up(float input, unsigned int lane_delta, int width=warpSize);
__device__ float __shfl_down(float input, unsigned int lane_delta, int width=warpSize);
__device__ float __shfl_xor(float input, int lane_mask, int width=warpSize);
#else
__device__ int __shfl(int input, int lane, int width);
__device__ int __shfl_up(int input, unsigned int lane_delta, int width);
__device__ int __shfl_down(int input, unsigned int lane_delta, int width);
__device__ int __shfl_xor(int input, int lane_mask, int width);
__device__ float __shfl(float input, int lane, int width);
__device__ float __shfl_up(float input, unsigned int lane_delta, int width);
__device__ float __shfl_down(float input, unsigned int lane_delta, int width);
__device__ float __shfl_xor(float input, int lane_mask, int width);
#endif

#include <hc_math.hpp>
// TODO: Choose whether default is precise math or fast math based on compilation flag.
#ifdef __HCC_ACCELERATOR__
using namespace hc::precise_math;
#endif

//TODO: Undo this once min/max functions are supported by hc
inline int min(int arg1, int arg2) __attribute((hc,cpu)) { \
  return (int)(hc::precise_math::fmin((float)arg1, (float)arg2));}
inline int max(int arg1, int arg2) __attribute((hc,cpu)) { \
  return (int)(hc::precise_math::fmax((float)arg1, (float)arg2));}


//TODO - add a couple fast math operations here, the set here will grow :
__device__ float __cosf(float x);
__device__ float __expf(float x);
__device__ float __frsqrt_rn(float x);
__device__ float __fsqrt_rd(float x);
__device__ float __fsqrt_rn(float x);
__device__ float __fsqrt_ru(float x);
__device__ float __fsqrt_rz(float x);
__device__ float __log10f(float x);
__device__ float __log2f(float x);
__device__ float __logf(float x);
__device__ float __powf(float base, float exponent);
__device__ void __sincosf(float x, float *s, float *c) ;
__device__ float __sinf(float x);
__device__ float __tanf(float x);
__device__ float __dsqrt_rd(double x);
__device__ float __dsqrt_rn(double x);
__device__ float __dsqrt_ru(double x);
__device__ float __dsqrt_rz(double x);

/**
 * Kernel launching
 */

#if __hcc_workweek__ >= 16123

#define hipThreadIdx_x (amp_get_local_id(0))
#define hipThreadIdx_y (amp_get_local_id(1))
#define hipThreadIdx_z (amp_get_local_id(2))

#define hipBlockIdx_x  (hc_get_group_id(0))
#define hipBlockIdx_y  (hc_get_group_id(1))
#define hipBlockIdx_z  (hc_get_group_id(2))

#define hipBlockDim_x  (amp_get_local_size(0))
#define hipBlockDim_y  (amp_get_local_size(1))
#define hipBlockDim_z  (amp_get_local_size(2))

#define hipGridDim_x   (hc_get_num_groups(0))
#define hipGridDim_y   (hc_get_num_groups(1))
#define hipGridDim_z   (hc_get_num_groups(2))

#else

#define hipThreadIdx_x (amp_get_local_id(2))
#define hipThreadIdx_y (amp_get_local_id(1))
#define hipThreadIdx_z (amp_get_local_id(0))

#define hipBlockIdx_x  (hc_get_group_id(2))
#define hipBlockIdx_y  (hc_get_group_id(1))
#define hipBlockIdx_z  (hc_get_group_id(0))

#define hipBlockDim_x  (amp_get_local_size(2))
#define hipBlockDim_y  (amp_get_local_size(1))
#define hipBlockDim_z  (amp_get_local_size(0))

#define hipGridDim_x   (hc_get_num_groups(2))
#define hipGridDim_y   (hc_get_num_groups(1))
#define hipGridDim_z   (hc_get_num_groups(0))

#endif


#define __syncthreads() hc_barrier(CLK_LOCAL_MEM_FENCE)

#define HIP_KERNEL_NAME(...) __VA_ARGS__

#ifdef __HCC_CPP__
hipStream_t ihipPreLaunchKernel(hipStream_t stream, hc::accelerator_view **av);
void ihipPostLaunchKernel(hipStream_t stream, hc::completion_future &cf);

// TODO - move to common header file.
#define KNRM  "\x1B[0m"
#define KGRN  "\x1B[32m"

#if not defined(DISABLE_GRID_LAUNCH)
#define hipLaunchKernel(_kernelName, _numBlocks3D, _blockDim3D, _groupMemBytes, _stream, ...) \
do {\
  grid_launch_parm lp;\
  lp.gridDim.x = _numBlocks3D.x; \
  lp.gridDim.y = _numBlocks3D.y; \
  lp.gridDim.z = _numBlocks3D.z; \
  lp.groupDim.x = _blockDim3D.x; \
  lp.groupDim.y = _blockDim3D.y; \
  lp.groupDim.z = _blockDim3D.z; \
  lp.groupMemBytes = _groupMemBytes;\
  hc::completion_future cf;\
  lp.cf = &cf;  \
  hipStream_t trueStream = (ihipPreLaunchKernel(_stream, &lp.av)); \
    if (HIP_TRACE_API) {\
        fprintf(stderr, KGRN "<<hip-api: hipLaunchKernel '%s' gridDim:[%d.%d.%d] groupDim:[%d.%d.%d] groupMem:+%d stream=%p\n" KNRM, \
                #_kernelName, lp.gridDim.z, lp.gridDim.y, lp.gridDim.x, lp.groupDim.z, lp.groupDim.y, lp.groupDim.x, lp.groupMemBytes, (void*)(_stream));\
    }\
  _kernelName (lp, __VA_ARGS__);\
  ihipPostLaunchKernel(trueStream, cf);\
} while(0)

#else
#warning(DISABLE_GRID_LAUNCH set)

#define hipLaunchKernel(_kernelName, _numBlocks3D, _blockDim3D, _groupMemBytes, _stream, ...) \
do {\
  grid_launch_parm lp;\
  lp.gridDim.x = _numBlocks3D.x * _blockDim3D.x;/*Convert from #blocks to #threads*/ \
  lp.gridDim.y = _numBlocks3D.y * _blockDim3D.y;/*Convert from #blocks to #threads*/ \
  lp.gridDim.z = _numBlocks3D.z * _blockDim3D.z;/*Convert from #blocks to #threads*/ \
  lp.groupDim.x = _blockDim3D.x; \
  lp.groupDim.y = _blockDim3D.y; \
  lp.groupDim.z = _blockDim3D.z; \
  lp.groupMemBytes = _groupMemBytes;\
  hc::completion_future cf;\
  lp.cf = &cf;  \
  hipStream_t trueStream = (ihipPreLaunchKernel(_stream, &lp.av)); \
    if (HIP_TRACE_API) {\
        fprintf(stderr, "==hip-api: launch '%s' gridDim:[%d.%d.%d] groupDim:[%d.%d.%d] groupMem:+%d stream=%p\n", \
                #_kernelName, lp.gridDim.z, lp.gridDim.y, lp.gridDim.x, lp.groupDim.z, lp.groupDim.y, lp.groupDim.x, lp.groupMemBytes, (void*)(_stream));\
    }\
  _kernelName (lp, __VA_ARGS__);\
  ihipPostLaunchKernel(trueStream, cf);\
} while(0)
/*end hipLaunchKernel */
#endif

#elif defined (__HCC_C__)

//TODO - develop C interface.

#endif

#endif // __HCC__


/**
 * @defgroup HIP-ENV HIP Environment Variables
 * @{
 */
//extern int HIP_PRINT_ENV ;   ///< Print all HIP-related environment variables.
//extern int HIP_TRACE_API;    ///< Trace HIP APIs.
//extern int HIP_LAUNCH_BLOCKING ; ///< Make all HIP APIs host-synchronous

/**
 * @}
 */


// End doxygen API:
/**
 *   @}
 */



