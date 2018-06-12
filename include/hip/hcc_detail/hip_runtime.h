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

#if defined(__HCC__)
#define __HCC_OR_HIP_CLANG__ 1
#define __HCC_ONLY__ 1
#define __HIP_CLANG_ONLY__ 0
#elif defined(__clang__) && defined(__HIP__)
#define __HCC_OR_HIP_CLANG__ 1
#define __HCC_ONLY__ 0
#define __HIP_CLANG_ONLY__ 1
#else
#define __HCC_OR_HIP_CLANG__ 0
#define __HCC_ONLY__ 0
#define __HIP_CLANG_ONLY__ 0
#endif

//---
// Top part of file can be compiled with any compiler

//#include <cstring>
#if __cplusplus
#include <cmath>
#else
#include <math.h>
#include <string.h>
#include <stddef.h>
#endif  //__cplusplus

#if __HCC_OR_HIP_CLANG__

// Define NVCC_COMPAT for CUDA compatibility
#define NVCC_COMPAT
#define CUDA_SUCCESS hipSuccess

#include <hip/hip_runtime_api.h>
#endif  // __HCC_OR_HIP_CLANG__

#if __HCC__
// define HIP_ENABLE_PRINTF to enable printf
#ifdef HIP_ENABLE_PRINTF
#define HCC_ENABLE_ACCELERATOR_PRINTF 1
#endif

//---
// Remainder of this file only compiles with HCC
#if defined __HCC__
#include <grid_launch.h>
#include "hc_printf.hpp"
// TODO-HCC-GL - change this to typedef.
// typedef grid_launch_parm hipLaunchParm ;

#if GENERIC_GRID_LAUNCH == 0
#define hipLaunchParm grid_launch_parm
#else
namespace hip_impl {
struct Empty_launch_parm {};
}  // namespace hip_impl
#define hipLaunchParm hip_impl::Empty_launch_parm
#endif  // GENERIC_GRID_LAUNCH

#if defined(GRID_LAUNCH_VERSION) and (GRID_LAUNCH_VERSION >= 20) || GENERIC_GRID_LAUNCH == 1
#else  // Use field names for grid_launch 2.0 structure, if HCC supports GL 2.0.
#error(HCC must support GRID_LAUNCH_20)
#endif  // GRID_LAUNCH_VERSION

#endif  // HCC

#if GENERIC_GRID_LAUNCH == 1 && defined __HCC__
#include "grid_launch_GGL.hpp"
#endif  // GENERIC_GRID_LAUNCH

#endif // HCC

#if __HCC_OR_HIP_CLANG__
extern int HIP_TRACE_API;

#ifdef __cplusplus
#include <hip/hcc_detail/hip_ldg.h>
#endif
#include <hip/hcc_detail/hip_atomic.h>
#include <hip/hcc_detail/host_defines.h>
#include <hip/hcc_detail/math_functions.h>
#include <hip/hcc_detail/device_functions.h>
#if __HCC___
#include <hip/hcc_detail/texture_functions.h>
#include <hip/hcc_detail/surface_functions.h>
#endif // __HCC__

// TODO-HCC remove old definitions ; ~1602 hcc supports __HCC_ACCELERATOR__ define.
#if defined(__KALMAR_ACCELERATOR__) && !defined(__HCC_ACCELERATOR__)
#define __HCC_ACCELERATOR__ __KALMAR_ACCELERATOR__
#endif

// TODO-HCC add a dummy implementation of assert, need to replace with a proper kernel exit call.
#if __HIP_DEVICE_COMPILE__ == 1
#undef assert
#define assert(COND)                                                                               \
    {                                                                                              \
        if (!(COND)) {                                                                             \
            abort();                                                                               \
        }                                                                                          \
    }
#endif


// Feature tests:
#if defined(__HCC_ACCELERATOR__) && (__HCC_ACCELERATOR__ != 0)
// Device compile and not host compile:

// 32-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ (1)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ (1)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ (1)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ (1)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ (1)

// 64-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ (1)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ (0)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__ (1)

// warp cross-lane operations:
#define __HIP_ARCH_HAS_WARP_VOTE__ (1)
#define __HIP_ARCH_HAS_WARP_BALLOT__ (1)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__ (1)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ (0)

// sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ (1)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__ (0)

// misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__ (0)
#define __HIP_ARCH_HAS_3DGRID__ (1)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ (0)

#endif /* Device feature flags */


#define launch_bounds_impl0(requiredMaxThreadsPerBlock)                                            \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define launch_bounds_impl1(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)                \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),                     \
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define select_impl_(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...)                                                                     \
    select_impl_(__VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0)(__VA_ARGS__)

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
__device__
unsigned long __llvm_amdgcn_s_memrealtime(void) __asm("llvm.amdgcn.s.memrealtime");

__device__
inline
long long int __clock64() { return (long long int)__llvm_amdgcn_s_memrealtime(); }

__device__
inline
clock_t  __clock() { return (clock_t)__llvm_amdgcn_s_memrealtime(); }

// abort
__device__ void abort();

// warp vote function __all __any __ballot
__device__ int __all(int input);
__device__ int __any(int input);
__device__ unsigned long long int __ballot(int input);

__device__
inline
int64_t __ballot64(int a) {
    int64_t s;
    // define i64 @__ballot64(i32 %a) #0 {
    //   %b = tail call i64 asm "v_cmp_ne_i32_e64 $0, 0, $1", "=s,v"(i32 %a) #1
    //   ret i64 %b
    // }
    __asm("v_cmp_ne_i32_e64 $0, 0, $1" : "=s"(s) : "v"(a));
    return s;
}

// hip.amdgcn.bc - lanemask
extern "C" __device__ int32_t __ockl_activelane_u32(void);

__device__
inline
int64_t __lanemask_gt()
{
    int32_t activelane = __ockl_activelane_u32();
    int64_t ballot = __ballot64(1);
    if (activelane != 63) {
        int64_t tmp = (~0UL) << (activelane + 1);
        return tmp & ballot;
    }
    return 0;
}

__device__
inline
int64_t __lanemask_lt()
{
    int32_t activelane = __ockl_activelane_u32();
    int64_t ballot = __ballot64(1);
    if (activelane == 0)
        return 0;
    return ballot;
}

#if __HIP_ARCH_GFX701__ == 0

// warp shuffle functions
#ifdef __cplusplus
__device__ int __shfl(int input, int lane, int width = warpSize);
__device__ int __shfl_up(int input, unsigned int lane_delta, int width = warpSize);
__device__ int __shfl_down(int input, unsigned int lane_delta, int width = warpSize);
__device__ int __shfl_xor(int input, int lane_mask, int width = warpSize);
__device__ float __shfl(float input, int lane, int width = warpSize);
__device__ float __shfl_up(float input, unsigned int lane_delta, int width = warpSize);
__device__ float __shfl_down(float input, unsigned int lane_delta, int width = warpSize);
__device__ float __shfl_xor(float input, int lane_mask, int width = warpSize);
#else
__device__ int __shfl(int input, int lane, int width);
__device__ int __shfl_up(int input, unsigned int lane_delta, int width);
__device__ int __shfl_down(int input, unsigned int lane_delta, int width);
__device__ int __shfl_xor(int input, int lane_mask, int width);
__device__ float __shfl(float input, int lane, int width);
__device__ float __shfl_up(float input, unsigned int lane_delta, int width);
__device__ float __shfl_down(float input, unsigned int lane_delta, int width);
__device__ float __shfl_xor(float input, int lane_mask, int width);
#endif  //__cplusplus

__device__ unsigned __hip_ds_bpermute(int index, unsigned src);
__device__ float __hip_ds_bpermutef(int index, float src);
__device__ unsigned __hip_ds_permute(int index, unsigned src);
__device__ float __hip_ds_permutef(int index, float src);

__device__ unsigned __hip_ds_swizzle(unsigned int src, int pattern);
__device__ float __hip_ds_swizzlef(float src, int pattern);

__device__ int __hip_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl);

#endif  //__HIP_ARCH_GFX803__ == 1

__host__ __device__ int min(int arg1, int arg2);
__host__ __device__ int max(int arg1, int arg2);

extern "C" __device__ void* get_dynamic_group_segment_base_pointer();

__device__
inline
void* __get_dynamicgroupbaseptr() { return get_dynamic_group_segment_base_pointer(); }

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
 *  If any of those threadfence stubs are reached by the application, you should set "export
 *HSA_DISABLE_CACHE=1" to disable L1 and L2 caches.
 *
 *
 *  On AMD platforms, the threadfence* routines are currently empty stubs.
 */

extern __attribute__((const)) __device__ void __hip_hc_threadfence() __asm("__llvm_fence_sc_dev");
extern __attribute__((const)) __device__ void __hip_hc_threadfence_block() __asm(
    "__llvm_fence_sc_wg");


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
__device__ static inline void __threadfence_block(void) { return __hip_hc_threadfence_block(); }

/**
 * @brief threadfence makes wirtes visible to other threads running on same GPU.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence is a stub and map to no-op, application should set "export
 * HSA_DISABLE_CACHE=1" to disable both L1 and L2 caches.
 */
// __device__ void  __threadfence(void) __attribute__((deprecated("Provided for compile-time
// compatibility, not yet functional")));
__device__ static inline void __threadfence(void) { return __hip_hc_threadfence(); }

/**
 * @brief threadfence_system makes writes to pinned system memory visible on host CPU.
 *
 * @Returns void
 *
 * @param void
 *
 * @warning __threadfence_system is a stub and map to no-op.
 */
//__device__ void  __threadfence_system(void) __attribute__((deprecated("Provided with workaround
//configuration, see hip_kernel_language.md for details")));
__device__ void __threadfence_system(void);

// doxygen end Fence Fence
/**
 * @}
 */

// hip.amdgcn.bc - named sync
__device__ void __llvm_amdgcn_s_barrier() __asm("llvm.amdgcn.s.barrier");

__device__ inline void __named_sync(int a, int b) { __llvm_amdgcn_s_barrier(); }

#endif  // __HCC_OR_HIP_CLANG__

#if defined __HCC__

template <
    typename std::common_type<decltype(hc_get_group_id), decltype(hc_get_group_size),
                              decltype(hc_get_num_groups), decltype(hc_get_workitem_id)>::type f>
class Coordinates {
    using R = decltype(f(0));

    struct X {
        __device__ operator R() const { return f(0); }
    };
    struct Y {
        __device__ operator R() const { return f(1); }
    };
    struct Z {
        __device__ operator R() const { return f(2); }
    };

   public:
    static constexpr X x{};
    static constexpr Y y{};
    static constexpr Z z{};
};

static constexpr Coordinates<hc_get_group_size> blockDim;
static constexpr Coordinates<hc_get_group_id> blockIdx;
static constexpr Coordinates<hc_get_num_groups> gridDim;
static constexpr Coordinates<hc_get_workitem_id> threadIdx;

#define hipThreadIdx_x (hc_get_workitem_id(0))
#define hipThreadIdx_y (hc_get_workitem_id(1))
#define hipThreadIdx_z (hc_get_workitem_id(2))

#define hipBlockIdx_x (hc_get_group_id(0))
#define hipBlockIdx_y (hc_get_group_id(1))
#define hipBlockIdx_z (hc_get_group_id(2))

#define hipBlockDim_x (hc_get_group_size(0))
#define hipBlockDim_y (hc_get_group_size(1))
#define hipBlockDim_z (hc_get_group_size(2))

#define hipGridDim_x (hc_get_num_groups(0))
#define hipGridDim_y (hc_get_num_groups(1))
#define hipGridDim_z (hc_get_num_groups(2))

#endif // defined __HCC__
#if __HCC_OR_HIP_CLANG__
extern "C" __device__ void* __hip_hc_memcpy(void* dst, const void* src, size_t size);
extern "C" __device__ void* __hip_hc_memset(void* ptr, uint8_t val, size_t size);
extern "C" __device__ void* __hip_hc_malloc(size_t);
extern "C" __device__ void* __hip_hc_free(void* ptr);

static inline __device__ void* malloc(size_t size) { return __hip_hc_malloc(size); }

static inline __device__ void* free(void* ptr) { return __hip_hc_free(ptr); }

static inline __device__ void* memcpy(void* dst, const void* src, size_t size) {
    return __hip_hc_memcpy(dst, src, size);
}

static inline __device__ void* memset(void* ptr, int val, size_t size) {
    uint8_t val8 = static_cast<uint8_t>(val);
    return __hip_hc_memset(ptr, val8, size);
}


#ifdef __HCC_ACCELERATOR__

#ifdef HC_FEATURE_PRINTF
template <typename... All>
static inline __device__ void printf(const char* format, All... all) {
    hc::printf(format, all...);
}
#else
template <typename... All>
static inline __device__ void printf(const char* format, All... all) {}
#endif

#endif
#endif //__HCC_OR_HIP_CLANG__

#ifdef __HCC__

#define __syncthreads() hc_barrier(CLK_LOCAL_MEM_FENCE)

#define HIP_KERNEL_NAME(...) (__VA_ARGS__)
#define HIP_SYMBOL(X) #X

#if defined __HCC_CPP__
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, dim3 block,
                                       grid_launch_parm* lp, const char* kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, size_t block,
                                       grid_launch_parm* lp, const char* kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, dim3 block,
                                       grid_launch_parm* lp, const char* kernelNameStr);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, size_t block,
                                       grid_launch_parm* lp, const char* kernelNameStr);
extern void ihipPostLaunchKernel(const char* kernelName, hipStream_t stream, grid_launch_parm& lp);

#if GENERIC_GRID_LAUNCH == 0
//#warning "Original hipLaunchKernel defined"
// Due to multiple overloaded versions of ihipPreLaunchKernel, the numBlocks3D and blockDim3D can be
// either size_t or dim3 types
#define hipLaunchKernel(_kernelName, _numBlocks3D, _blockDim3D, _groupMemBytes, _stream, ...)      \
    do {                                                                                           \
        grid_launch_parm lp;                                                                       \
        lp.dynamic_group_mem_bytes = _groupMemBytes;                                               \
        hipStream_t trueStream =                                                                   \
            (ihipPreLaunchKernel(_stream, _numBlocks3D, _blockDim3D, &lp, #_kernelName));          \
        _kernelName(lp, ##__VA_ARGS__);                                                            \
        ihipPostLaunchKernel(#_kernelName, trueStream, lp);                                        \
    } while (0)
#endif  // GENERIC_GRID_LAUNCH

#elif defined(__HCC_C__)

// TODO - develop C interface.

#endif  //__HCC_CPP__

/**
 * extern __shared__
 */

// Macro to replace extern __shared__ declarations
// to local variable definitions
#define HIP_DYNAMIC_SHARED(type, var) type* var = (type*)__get_dynamicgroupbaseptr();

#define HIP_DYNAMIC_SHARED_ATTRIBUTE


/**
 * @defgroup HIP-ENV HIP Environment Variables
 * @{
 */
// extern int HIP_PRINT_ENV ;   ///< Print all HIP-related environment variables.
// extern int HIP_TRACE_API;    ///< Trace HIP APIs.
// extern int HIP_LAUNCH_BLOCKING ; ///< Make all HIP APIs host-synchronous

/**
 * @}
 */


// End doxygen API:
/**
 *   @}
 */

//
// hip-clang functions
//
#elif defined(__clang__) && defined(__HIP__)

#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define HIP_SYMBOL(X) #X

typedef int hipLaunchParm;

#define hipLaunchKernel(kernelName, numblocks, numthreads, memperblock, streamId, ...)             \
    do {                                                                                           \
        kernelName<<<numblocks, numthreads, memperblock, streamId>>>(0, ##__VA_ARGS__);            \
    } while (0)

#define hipLaunchKernelGGL(kernelName, numblocks, numthreads, memperblock, streamId, ...)          \
    do {                                                                                           \
        kernelName<<<numblocks, numthreads, memperblock, streamId>>>(__VA_ARGS__);                 \
    } while (0)

#include <hip/hip_runtime_api.h>

#pragma push_macro("__DEVICE__")
#define __DEVICE__ static __device__ __forceinline__

extern "C" __device__ size_t __ockl_get_local_id(uint);
__DEVICE__ uint __hip_get_thread_idx_x() { return __ockl_get_local_id(0); }
__DEVICE__ uint __hip_get_thread_idx_y() { return __ockl_get_local_id(1); }
__DEVICE__ uint __hip_get_thread_idx_z() { return __ockl_get_local_id(2); }

extern "C" __device__ size_t __ockl_get_group_id(uint);
__DEVICE__ uint __hip_get_block_idx_x() { return __ockl_get_group_id(0); }
__DEVICE__ uint __hip_get_block_idx_y() { return __ockl_get_group_id(1); }
__DEVICE__ uint __hip_get_block_idx_z() { return __ockl_get_group_id(2); }

extern "C" __device__ size_t __ockl_get_local_size(uint);
__DEVICE__ uint __hip_get_block_dim_x() { return __ockl_get_local_size(0); }
__DEVICE__ uint __hip_get_block_dim_y() { return __ockl_get_local_size(1); }
__DEVICE__ uint __hip_get_block_dim_z() { return __ockl_get_local_size(2); }

extern "C" __device__ size_t __ockl_get_num_groups(uint);
__DEVICE__ uint __hip_get_grid_dim_x() { return __ockl_get_num_groups(0); }
__DEVICE__ uint __hip_get_grid_dim_y() { return __ockl_get_num_groups(1); }
__DEVICE__ uint __hip_get_grid_dim_z() { return __ockl_get_num_groups(2); }

#define __HIP_DEVICE_BUILTIN(DIMENSION, FUNCTION)               \
  __declspec(property(get = __get_##DIMENSION)) uint DIMENSION; \
  __DEVICE__ uint __get_##DIMENSION(void) {                     \
    return FUNCTION;                                            \
  }

struct __hip_builtin_threadIdx_t {
  __HIP_DEVICE_BUILTIN(x,__hip_get_thread_idx_x());
  __HIP_DEVICE_BUILTIN(y,__hip_get_thread_idx_y());
  __HIP_DEVICE_BUILTIN(z,__hip_get_thread_idx_z());
};

struct __hip_builtin_blockIdx_t {
  __HIP_DEVICE_BUILTIN(x,__hip_get_block_idx_x());
  __HIP_DEVICE_BUILTIN(y,__hip_get_block_idx_y());
  __HIP_DEVICE_BUILTIN(z,__hip_get_block_idx_z());
};

struct __hip_builtin_blockDim_t {
  __HIP_DEVICE_BUILTIN(x,__hip_get_block_dim_x());
  __HIP_DEVICE_BUILTIN(y,__hip_get_block_dim_y());
  __HIP_DEVICE_BUILTIN(z,__hip_get_block_dim_z());
};

struct __hip_builtin_gridDim_t {
  __HIP_DEVICE_BUILTIN(x,__hip_get_grid_dim_x());
  __HIP_DEVICE_BUILTIN(y,__hip_get_grid_dim_y());
  __HIP_DEVICE_BUILTIN(z,__hip_get_grid_dim_z());
};

#undef __HIP_DEVICE_BUILTIN
#pragma pop_macro("__DEVICE__")

extern const __device__ __attribute__((weak)) __hip_builtin_threadIdx_t threadIdx;
extern const __device__ __attribute__((weak)) __hip_builtin_blockIdx_t blockIdx;
extern const __device__ __attribute__((weak)) __hip_builtin_blockDim_t blockDim;
extern const __device__ __attribute__((weak)) __hip_builtin_gridDim_t gridDim;


#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z

#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y
#define hipGridDim_z gridDim.z

#pragma push_macro("__DEVICE__")
#define __DEVICE__ extern "C" __device__ __attribute__((always_inline)) \
  __attribute__((weak))

__DEVICE__ void __device_trap() __asm("llvm.trap");

__DEVICE__
inline
void __assert_fail(const char * __assertion,
                                     const char *__file,
                                     unsigned int __line,
                                     const char *__function)
{
    // Ignore all the args for now.
    __device_trap();
}

__DEVICE__
inline
void __assertfail(const char * __assertion,
                  const char *__file,
                  unsigned int __line,
                  const char *__function,
                  size_t charsize)
{
    // ignore all the args for now.
    __device_trap();
}

// hip.amdgcn.bc - sync threads
// extern "C" __device__ __attribute__((noduplicate)) void __syncthreads();
#define CLK_LOCAL_MEM_FENCE    0x01
#define local __attribute__((address_space(3)))

typedef unsigned cl_mem_fence_flags;

typedef enum memory_scope {
  memory_scope_work_item = __OPENCL_MEMORY_SCOPE_WORK_ITEM,
  memory_scope_work_group = __OPENCL_MEMORY_SCOPE_WORK_GROUP,
  memory_scope_device = __OPENCL_MEMORY_SCOPE_DEVICE,
  memory_scope_all_svm_devices = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
  memory_scope_sub_group = __OPENCL_MEMORY_SCOPE_SUB_GROUP
} memory_scope;

// enum values aligned with what clang uses in EmitAtomicExpr()
typedef enum memory_order
{
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

extern "C" __device__ __attribute__((overloadable))
void atomic_work_item_fence(cl_mem_fence_flags, memory_order, memory_scope);

__device__
inline
static void hc_work_group_barrier(cl_mem_fence_flags flags, memory_scope scope)
{
    if (flags) {
        atomic_work_item_fence(flags, memory_order_release, scope);
        __builtin_amdgcn_s_barrier();
        atomic_work_item_fence(flags, memory_order_acquire, scope);
    } else {
        __builtin_amdgcn_s_barrier();
    }
}

__device__
inline
static void hc_barrier(int n)
{
  hc_work_group_barrier((cl_mem_fence_flags)n, memory_scope_work_group);
}

__device__
inline
__attribute__((noduplicate))
void __syncthreads()
{
  hc_barrier(CLK_LOCAL_MEM_FENCE);
}


__device__ unsigned __llvm_amdgcn_s_getreg(unsigned) __asm("llvm.amdgcn.s.getreg");

__device__ unsigned __llvm_amdgcn_groupstaticsize() __asm("llvm.amdgcn.groupstaticsize");

__device__ inline static local char* __to_local(unsigned x) { return (local char*)x; }

__device__ inline void *__amdgcn_get_dynamicgroupbaseptr() {
#if 0
    // Get group segment base pointer.
    char* base = __llvm_amdgcn_s_getreg(14342) << 8);
    base += __llvm_amdgcn_groupstaticsize();
    return base;
#endif
    return __get_dynamicgroupbaseptr();
}

// hip.amdgcn.bc - device routine
/*
   HW_ID Register bit structure
   WAVE_ID     3:0     Wave buffer slot number. 0-9.
   SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
   PIPE_ID     7:6     Pipeline from which the wave was dispatched.
   CU_ID       11:8    Compute Unit the wave is assigned to.
   SH_ID       12      Shader Array (within an SE) the wave is assigned to.
   SE_ID       14:13   Shader Engine the wave is assigned to.
   TG_ID       19:16   Thread-group ID
   VM_ID       23:20   Virtual Memory ID
   QUEUE_ID    26:24   Queue from which this wave was dispatched.
   STATE_ID    29:27   State ID (graphics only, not compute).
   ME_ID       31:30   Micro-engine ID.
 */

#define HW_ID               4

#define HW_ID_CU_ID_SIZE    4
#define HW_ID_CU_ID_OFFSET  8

#define HW_ID_SE_ID_SIZE    2
#define HW_ID_SE_ID_OFFSET  13

/*
   Encoding of parameter bitmask
   HW_ID        5:0     HW_ID
   OFFSET       10:6    Range: 0..31
   SIZE         15:11   Range: 1..32
 */

#define GETREG_IMMED(SZ,OFF,REG) (SZ << 11) | (OFF << 6) | REG

__device__
inline
unsigned __smid(void)
{
    unsigned cu_id = __llvm_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE, HW_ID_CU_ID_OFFSET, HW_ID));
    unsigned se_id = __llvm_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SE_ID_SIZE, HW_ID_SE_ID_OFFSET, HW_ID));

    /* Each shader engine has 16 CU */
    return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
}

// Macro to replace extern __shared__ declarations
// to local variable definitions
#define HIP_DYNAMIC_SHARED(type, var) \
    type* var = (type*)__amdgcn_get_dynamicgroupbaseptr();

#define HIP_DYNAMIC_SHARED_ATTRIBUTE

#pragma push_macro("__DEVICE__")

#include <hip/hcc_detail/math_functions.h>

#endif

#endif  // HIP_HCC_DETAIL_RUNTIME_H
