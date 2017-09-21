/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

//#pragma once
#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_H

//---
// Top part of file can be compiled with any compiler

//#include <cstring>
#if __cplusplus
#include <cmath>
#else
#include <math.h>
#include <string.h>
#include <stddef.h>
#endif//__cplusplus

#if __HCC__

// Define NVCC_COMPAT for CUDA compatibility
#define NVCC_COMPAT
#define CUDA_SUCCESS hipSuccess

#include <hip/hip_runtime_api.h>


//---
// Remainder of this file only compiles with HCC
#if defined __HCC__
#include <grid_launch.h>
//TODO-HCC-GL - change this to typedef.
//typedef grid_launch_parm hipLaunchParm ;

#if GENERIC_GRID_LAUNCH == 0
    #define hipLaunchParm grid_launch_parm
#else
namespace hip_impl
{
    struct Empty_launch_parm {};
}
#define hipLaunchParm hip_impl::Empty_launch_parm
#endif //GENERIC_GRID_LAUNCH

#if defined (GRID_LAUNCH_VERSION) and (GRID_LAUNCH_VERSION >= 20) || GENERIC_GRID_LAUNCH == 1
#else // Use field names for grid_launch 2.0 structure, if HCC supports GL 2.0.
#error (HCC must support GRID_LAUNCH_20)
#endif //GRID_LAUNCH_VERSION

#endif //HCC

#if GENERIC_GRID_LAUNCH==1 && defined __HCC__
#include "grid_launch_GGL.hpp"
#endif//GENERIC_GRID_LAUNCH

extern int HIP_TRACE_API;

#ifdef __cplusplus
#include <hip/hcc_detail/hip_ldg.h>
#endif
#include <hip/hcc_detail/host_defines.h>
#include <hip/hcc_detail/math_functions.h>
#include <hip/hcc_detail/device_functions.h>
#include <hip/hcc_detail/texture_functions.h>

// TODO-HCC remove old definitions ; ~1602 hcc supports __HCC_ACCELERATOR__ define.
#if defined (__KALMAR_ACCELERATOR__) && !defined (__HCC_ACCELERATOR__)
#define __HCC_ACCELERATOR__  __KALMAR_ACCELERATOR__
#endif




// TODO-HCC add a dummy implementation of assert, need to replace with a proper kernel exit call.
#if __HIP_DEVICE_COMPILE__ == 1
   #undef assert
   #define assert(COND) { if (!(COND)) {abort();} }
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
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__        (1)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__            (0)

// misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__              (0)
#define __HIP_ARCH_HAS_3DGRID__                     (1)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__           (0)

#endif /* Device feature flags */


#define launch_bounds_impl0(requiredMaxThreadsPerBlock)\
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define launch_bounds_impl1(\
    requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)\
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),\
        amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define select_impl_(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...) select_impl_(\
    __VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0)(__VA_ARGS__)

// Detect if we are compiling C++ mode or C mode
#if defined(__cplusplus)
#define __HCC_CPP__
#elif defined(__STDC_VERSION__)
#define __HCC_C__
#endif

// TODO - hipify-clang - change to use the function call.
//#define warpSize hc::__wavesize()
static constexpr int warpSize = 64;

#define clock_t long long int
__device__ long long int clock64();
__device__ clock_t clock();

//abort
__device__ void abort();

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

//atomicInc()
__device__ unsigned int atomicInc(unsigned int* address,
                       unsigned int val);


//atomicDec()
__device__ unsigned int atomicDec(unsigned int* address,
                       unsigned int val);

                       // warp vote function __all __any __ballot
__device__ int __all(  int input);
__device__ int __any( int input);
__device__  unsigned long long int __ballot( int input);

#if __HIP_ARCH_GFX701__ == 0

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
#endif //__cplusplus

__device__ unsigned __hip_ds_bpermute(int index, unsigned src);
__device__ float __hip_ds_bpermutef(int index, float src);
__device__ unsigned __hip_ds_permute(int index, unsigned src);
__device__ float __hip_ds_permutef(int index, float src);

__device__ unsigned __hip_ds_swizzle(unsigned int src, int pattern);
__device__ float __hip_ds_swizzlef(float src, int pattern);

__device__ int __hip_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl);

#endif //__HIP_ARCH_GFX803__ == 1

__host__ __device__ int min(int arg1, int arg2);
__host__ __device__ int max(int arg1, int arg2);

__device__ void* __get_dynamicgroupbaseptr();


/**
 * CUDA 8 device function features

 */


/**
 * Kernel launching
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Fence Fence Functions
 *  @{
 *
 *
 *  @warning The HIP memory fence functions are currently not supported yet.
 *  If any of those threadfence stubs are reached by the application, you should set "export HSA_DISABLE_CACHE=1" to disable L1 and L2 caches.
 *
 *
 *  On AMD platforms, the threadfence* routines are currently empty stubs.
 */

extern __attribute__((const)) __device__ void __hip_hc_threadfence() __asm("__llvm_fence_sc_dev");
extern __attribute__((const)) __device__ void __hip_hc_threadfence_block() __asm("__llvm_fence_sc_wg");


 /**
 * @brief threadfence_block makes writes visible to threads running in same block.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence_block is a stub and map to no-op.
 */
// __device__ void  __threadfence_block(void);
__device__ static inline void __threadfence_block(void) {
  return __hip_hc_threadfence_block();
}

 /**
  * @brief threadfence makes wirtes visible to other threads running on same GPU.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence is a stub and map to no-op, application should set "export HSA_DISABLE_CACHE=1" to disable both L1 and L2 caches.
 */
// __device__ void  __threadfence(void) __attribute__((deprecated("Provided for compile-time compatibility, not yet functional")));
__device__ static inline void __threadfence(void) {
  return __hip_hc_threadfence();
}

/**
 * @brief threadfence_system makes writes to pinned system memory visible on host CPU.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence_system is a stub and map to no-op.
 */
//__device__ void  __threadfence_system(void) __attribute__((deprecated("Provided with workaround configuration, see hip_kernel_language.md for details")));
__device__ void  __threadfence_system(void) ;

// doxygen end Fence Fence
/**
 * @}
 */


#define hipThreadIdx_x (hc_get_workitem_id(0))
#define hipThreadIdx_y (hc_get_workitem_id(1))
#define hipThreadIdx_z (hc_get_workitem_id(2))

#define hipBlockIdx_x  (hc_get_group_id(0))
#define hipBlockIdx_y  (hc_get_group_id(1))
#define hipBlockIdx_z  (hc_get_group_id(2))

#define hipBlockDim_x  (hc_get_group_size(0))
#define hipBlockDim_y  (hc_get_group_size(1))
#define hipBlockDim_z  (hc_get_group_size(2))

#define hipGridDim_x   (hc_get_num_groups(0))
#define hipGridDim_y   (hc_get_num_groups(1))
#define hipGridDim_z   (hc_get_num_groups(2))

extern "C" __device__ void* __hip_hc_memcpy(void* dst, const void* src, size_t size);
extern "C" __device__ void* __hip_hc_memset(void* ptr, uint8_t val, size_t size);
extern "C" __device__ void* __hip_hc_malloc(size_t);
extern "C" __device__ void* __hip_hc_free(void *ptr);

static inline __device__ void* malloc(size_t size)
{
    return __hip_hc_malloc(size);
}

static inline __device__ void* free(void *ptr)
{
    return __hip_hc_free(ptr);
}

static inline __device__ void* memcpy(void* dst, const void* src, size_t size)
{
  return __hip_hc_memcpy(dst, src, size);
}

static inline __device__ void* memset(void* ptr, int val, size_t size)
{
  uint8_t val8 = static_cast <uint8_t> (val);
  return __hip_hc_memset(ptr, val8, size);
}



#define __syncthreads() hc_barrier(CLK_LOCAL_MEM_FENCE)

#define HIP_KERNEL_NAME(...)  (__VA_ARGS__)
#define HIP_SYMBOL(X) #X

#if defined __HCC_CPP__
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, dim3 block, grid_launch_parm *lp, const char *kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, size_t block, grid_launch_parm *lp, const char *kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, dim3 block, grid_launch_parm *lp, const char *kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, size_t block, grid_launch_parm *lp, const char *kernelNameStr);
extern void ihipPostLaunchKernel(const char *kernelName, hipStream_t stream, grid_launch_parm &lp);

#if GENERIC_GRID_LAUNCH == 0
//#warning "Original hipLaunchKernel defined"
// Due to multiple overloaded versions of ihipPreLaunchKernel, the numBlocks3D and blockDim3D can be either size_t or dim3 types
#define hipLaunchKernel(_kernelName, _numBlocks3D, _blockDim3D, _groupMemBytes, _stream, ...) \
do {\
  grid_launch_parm lp;\
  lp.dynamic_group_mem_bytes = _groupMemBytes; \
  hipStream_t trueStream = (ihipPreLaunchKernel(_stream, _numBlocks3D, _blockDim3D, &lp, #_kernelName)); \
  _kernelName (lp, ##__VA_ARGS__);\
  ihipPostLaunchKernel(#_kernelName, trueStream, lp);\
} while(0)
#endif //GENERIC_GRID_LAUNCH

#elif defined (__HCC_C__)

//TODO - develop C interface.

#endif //__HCC_CPP__

/**
 * extern __shared__
 */

// Macro to replace extern __shared__ declarations
// to local variable definitions
#define HIP_DYNAMIC_SHARED(type, var) \
    type* var = \
    (type*)__get_dynamicgroupbaseptr(); \

#define HIP_DYNAMIC_SHARED_ATTRIBUTE 



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


#endif

#endif//HIP_HCC_DETAIL_RUNTIME_H
