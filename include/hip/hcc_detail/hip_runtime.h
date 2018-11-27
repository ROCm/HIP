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
#include "grid_launch.h"
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
#include <hip/hcc_detail/device_functions.h>
#include <hip/hcc_detail/surface_functions.h>
#include <hip/hcc_detail/texture_functions.h>
#if __HCC__
    #include <hip/hcc_detail/math_functions.h>
#endif
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
#if (defined(__HCC_ACCELERATOR__) && (__HCC_ACCELERATOR__ != 0)) || __HIP_DEVICE_COMPILE__
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

__host__ inline void* __get_dynamicgroupbaseptr() { return nullptr; }

#if __HIP_ARCH_GFX701__ == 0

__device__ unsigned __hip_ds_bpermute(int index, unsigned src);
__device__ float __hip_ds_bpermutef(int index, float src);
__device__ unsigned __hip_ds_permute(int index, unsigned src);
__device__ float __hip_ds_permutef(int index, float src);

__device__ unsigned __hip_ds_swizzle(unsigned int src, int pattern);
__device__ float __hip_ds_swizzlef(float src, int pattern);

__device__ int __hip_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl);

#endif  //__HIP_ARCH_GFX803__ == 1

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
extern "C" __device__ void* __hip_malloc(size_t);
extern "C" __device__ void* __hip_free(void* ptr);

static inline __device__ void* malloc(size_t size) { return __hip_malloc(size); }
static inline __device__ void* free(void* ptr) { return __hip_free(ptr); }

#if defined(__HCC_ACCELERATOR__) && defined(HC_FEATURE_PRINTF)
template <typename... All>
static inline __device__ void printf(const char* format, All... all) {
    hc::printf(format, all...);
}
#elif defined(__HCC_ACCELERATOR__) || __HIP__
template <typename... All>
static inline __device__ void printf(const char* format, All... all) {}
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

template <typename... Args, typename F = void (*)(Args...)>
inline void hipLaunchKernelGGL(F&& kernelName, const dim3& numblocks, const dim3& numthreads,
                               unsigned memperblock, hipStream_t streamId, Args... args) {
  kernelName<<<numblocks, numthreads, memperblock, streamId>>>(args...);
}

template <typename... Args, typename F = void (*)(hipLaunchParm, Args...)>
inline void hipLaunchKernel(F&& kernel, const dim3& numBlocks, const dim3& dimBlocks,
                            std::uint32_t groupMemBytes, hipStream_t stream, Args... args) {
    hipLaunchKernelGGL(kernel, numBlocks, dimBlocks, groupMemBytes, stream, hipLaunchParm{},
                       std::move(args)...);
}

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

#include <hip/hcc_detail/math_functions.h>

#if __HIP_HCC_COMPAT_MODE__
// Define HCC work item functions in terms of HIP builtin variables.
#pragma push_macro("__DEFINE_HCC_FUNC")
#define __DEFINE_HCC_FUNC(hc_fun,hip_var) \
inline __device__ __attribute__((always_inline)) uint hc_get_##hc_fun(uint i) { \
  if (i==0) \
    return hip_var.x; \
  else if(i==1) \
    return hip_var.y; \
  else \
    return hip_var.z; \
}

__DEFINE_HCC_FUNC(workitem_id, threadIdx)
__DEFINE_HCC_FUNC(group_id, blockIdx)
__DEFINE_HCC_FUNC(group_size, blockDim)
__DEFINE_HCC_FUNC(num_groups, gridDim)
#pragma pop_macro("__DEFINE_HCC_FUNC")

extern "C" __device__ __attribute__((const)) size_t __ockl_get_global_id(uint);
inline __device__ __attribute__((always_inline)) uint
hc_get_workitem_absolute_id(int dim)
{
  return (uint)__ockl_get_global_id(dim);
}

#endif

// Support std::complex.
#pragma push_macro("__CUDA__")
#define __CUDA__
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <cuda_wrappers/algorithm>
#include <cuda_wrappers/complex>
#include <cuda_wrappers/new>
#undef __CUDA__
#pragma pop_macro("__CUDA__")


hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent = nullptr,
                                    hipEvent_t stopEvent = nullptr);

#endif // defined(__clang__) && defined(__HIP__)

#include <hip/hcc_detail/hip_memory.h>

#endif  // HIP_HCC_DETAIL_RUNTIME_H
