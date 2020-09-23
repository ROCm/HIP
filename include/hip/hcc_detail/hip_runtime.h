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

#include <hip/hcc_detail/hip_common.h>

//---
// Top part of file can be compiled with any compiler

//#include <cstring>
#if __cplusplus
#include <cmath>
#include <cstdint>
#else
#include <math.h>
#include <string.h>
#include <stddef.h>
#endif  //__cplusplus

// __hip_malloc is not working. Disable it by default.
#ifndef __HIP_ENABLE_DEVICE_MALLOC__
#define __HIP_ENABLE_DEVICE_MALLOC__ 0
#endif

#if __HCC_OR_HIP_CLANG__

#if __HIP__
#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif
#endif

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
#if __HCC__
    #include <hip/hcc_detail/math_functions.h>
    #include <hip/hcc_detail/texture_functions.h>
#else
    #include <hip/hcc_detail/texture_fetch_functions.h>
    #include <hip/hcc_detail/texture_indirect_functions.h>
#endif
// TODO-HCC remove old definitions ; ~1602 hcc supports __HCC_ACCELERATOR__ define.
#if defined(__KALMAR_ACCELERATOR__) && !defined(__HCC_ACCELERATOR__)
#define __HCC_ACCELERATOR__ __KALMAR_ACCELERATOR__
#endif

// TODO-HCC add a dummy implementation of assert, need to replace with a proper kernel exit call.
#if defined(__HCC__) && __HIP_DEVICE_COMPILE__ == 1
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

template <int pattern>
__device__ unsigned __hip_ds_swizzle_N(unsigned int src);
template <int pattern>
__device__ float __hip_ds_swizzlef_N(float src);

template <int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl>
__device__ int __hip_move_dpp_N(int src);

#endif  //__HIP_ARCH_GFX803__ == 1

#endif  // __HCC_OR_HIP_CLANG__

#if defined __HCC__

namespace hip_impl {
  struct GroupId {
    using R = decltype(hc_get_group_id(0));

    __device__
    R operator()(std::uint32_t x) const noexcept { return hc_get_group_id(x); }
  };
  struct GroupSize {
    using R = decltype(hc_get_group_size(0));

    __device__
    R operator()(std::uint32_t x) const noexcept {
      return hc_get_group_size(x);
    }
  };
  struct NumGroups {
    using R = decltype(hc_get_num_groups(0));

    __device__
    R operator()(std::uint32_t x) const noexcept {
      return hc_get_num_groups(x);
    }
  };
  struct WorkitemId {
    using R = decltype(hc_get_workitem_id(0));

    __device__
    R operator()(std::uint32_t x) const noexcept {
      return hc_get_workitem_id(x);
    }
  };
} // Namespace hip_impl.

template <typename F>
struct Coordinates {
  using R = decltype(F{}(0));

  struct X { __device__ operator R() const noexcept { return F{}(0); } };
  struct Y { __device__ operator R() const noexcept { return F{}(1); } };
  struct Z { __device__ operator R() const noexcept { return F{}(2); } };

  static constexpr X x{};
  static constexpr Y y{};
  static constexpr Z z{};
};

inline
__device__
std::uint32_t operator*(Coordinates<hip_impl::NumGroups>::X,
                        Coordinates<hip_impl::GroupSize>::X) noexcept {
  return hc_get_grid_size(0);
}
inline
__device__
std::uint32_t operator*(Coordinates<hip_impl::GroupSize>::X,
                        Coordinates<hip_impl::NumGroups>::X) noexcept {
  return hc_get_grid_size(0);
}
inline
__device__
std::uint32_t operator*(Coordinates<hip_impl::NumGroups>::Y,
                        Coordinates<hip_impl::GroupSize>::Y) noexcept {
  return hc_get_grid_size(1);
}
inline
__device__
std::uint32_t operator*(Coordinates<hip_impl::GroupSize>::Y,
                        Coordinates<hip_impl::NumGroups>::Y) noexcept {
  return hc_get_grid_size(1);
}
inline
__device__
std::uint32_t operator*(Coordinates<hip_impl::NumGroups>::Z,
                        Coordinates<hip_impl::GroupSize>::Z) noexcept {
  return hc_get_grid_size(2);
}
inline
__device__
std::uint32_t operator*(Coordinates<hip_impl::GroupSize>::Z,
                        Coordinates<hip_impl::NumGroups>::Z) noexcept {
  return hc_get_grid_size(2);
}

static constexpr Coordinates<hip_impl::GroupSize> blockDim{};
static constexpr Coordinates<hip_impl::GroupId> blockIdx{};
static constexpr Coordinates<hip_impl::NumGroups> gridDim{};
static constexpr Coordinates<hip_impl::WorkitemId> threadIdx{};

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
#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#if __HIP_ENABLE_DEVICE_MALLOC__
extern "C" __device__ void* __hip_malloc(size_t);
extern "C" __device__ void* __hip_free(void* ptr);
static inline __device__ void* malloc(size_t size) { return __hip_malloc(size); }
static inline __device__ void* free(void* ptr) { return __hip_free(ptr); }
#else
static inline __device__ void* malloc(size_t size) { __builtin_trap(); return nullptr; }
static inline __device__ void* free(void* ptr) { __builtin_trap(); return nullptr; }
#endif
#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#endif //__HCC_OR_HIP_CLANG__

#ifdef __HCC__

#define __syncthreads() hc_barrier(CLK_LOCAL_MEM_FENCE)

#define HIP_KERNEL_NAME(...) (__VA_ARGS__)
#define HIP_SYMBOL(X) #X

#if defined __HCC_CPP__
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, dim3 block,
                                       grid_launch_parm* lp, const char* kernelNameStr, bool lockAcquired = 0);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, dim3 grid, size_t block,
                                       grid_launch_parm* lp, const char* kernelNameStr, bool lockAcquired = 0);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, dim3 block,
                                       grid_launch_parm* lp, const char* kernelNameStr, bool lockAcquired = 0);
extern hipStream_t ihipPreLaunchKernel(hipStream_t stream, size_t grid, size_t block,
                                       grid_launch_parm* lp, const char* kernelNameStr, bool lockAcquired = 0);
extern void ihipPostLaunchKernel(const char* kernelName, hipStream_t stream, grid_launch_parm& lp, bool unlockPostponed = 0);

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
#define HIP_SYMBOL(X) X

typedef int hipLaunchParm;

template <std::size_t n, typename... Ts,
          typename std::enable_if<n == sizeof...(Ts)>::type* = nullptr>
void pArgs(const std::tuple<Ts...>&, void*) {}

template <std::size_t n, typename... Ts,
          typename std::enable_if<n != sizeof...(Ts)>::type* = nullptr>
void pArgs(const std::tuple<Ts...>& formals, void** _vargs) {
    using T = typename std::tuple_element<n, std::tuple<Ts...> >::type;

    static_assert(!std::is_reference<T>{},
                  "A __global__ function cannot have a reference as one of its "
                  "arguments.");
#if defined(HIP_STRICT)
    static_assert(std::is_trivially_copyable<T>{},
                  "Only TriviallyCopyable types can be arguments to a __global__ "
                  "function");
#endif
    _vargs[n] = const_cast<void*>(reinterpret_cast<const void*>(&std::get<n>(formals)));
    return pArgs<n + 1>(formals, _vargs);
}

template <typename... Formals, typename... Actuals>
std::tuple<Formals...> validateArgsCountType(void (*kernel)(Formals...), std::tuple<Actuals...>(actuals)) {
    static_assert(sizeof...(Formals) == sizeof...(Actuals), "Argument Count Mismatch");
    std::tuple<Formals...> to_formals{std::move(actuals)};
    return to_formals;
}

#if defined(HIP_TEMPLATE_KERNEL_LAUNCH)
template <typename... Args, typename F = void (*)(Args...)>
void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                        std::uint32_t sharedMemBytes, hipStream_t stream, Args... args) {
    constexpr size_t count = sizeof...(Args);
    auto tup_ = std::tuple<Args...>{args...};
    auto tup = validateArgsCountType(kernel, tup_);
    void* _Args[count];
    pArgs<0>(tup, _Args);

    auto k = reinterpret_cast<void*>(kernel);
    hipLaunchKernel(k, numBlocks, dimBlocks, _Args, sharedMemBytes, stream);
}
#else
#define hipLaunchKernelGGLInternal(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)  \
    do {                                                                                           \
        kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);         \
    } while (0)

#define hipLaunchKernelGGL(kernelName, ...)  hipLaunchKernelGGLInternal((kernelName), __VA_ARGS__)
#endif

#include <hip/hip_runtime_api.h>
extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_id(uint);
extern "C" __device__ __attribute__((const)) size_t __ockl_get_group_id(uint);
extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_size(uint);
extern "C" __device__ __attribute__((const)) size_t __ockl_get_num_groups(uint);
struct __HIP_BlockIdx {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept { return __ockl_get_group_id(x); }
};
struct __HIP_BlockDim {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_local_size(x);
  }
};
struct __HIP_GridDim {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_num_groups(x);
  }
};
struct __HIP_ThreadIdx {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_local_id(x);
  }
};

template <typename F>
struct __HIP_Coordinates {
  using R = decltype(F{}(0));

  struct X { __device__ operator R() const noexcept { return F{}(0); } };
  struct Y { __device__ operator R() const noexcept { return F{}(1); } };
  struct Z { __device__ operator R() const noexcept { return F{}(2); } };

  static constexpr X x{};
  static constexpr Y y{};
  static constexpr Z z{};
#ifdef __cplusplus
  __device__ operator dim3() const { return dim3(x, y, z); }
#endif

};
template <typename F>
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
constexpr typename __HIP_Coordinates<F>::X __HIP_Coordinates<F>::x;
template <typename F>
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
constexpr typename __HIP_Coordinates<F>::Y __HIP_Coordinates<F>::y;
template <typename F>
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
constexpr typename __HIP_Coordinates<F>::Z __HIP_Coordinates<F>::z;

extern "C" __device__ __attribute__((const)) size_t __ockl_get_global_size(uint);
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::X,
                        __HIP_Coordinates<__HIP_BlockDim>::X) noexcept {
  return __ockl_get_global_size(0);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::X,
                        __HIP_Coordinates<__HIP_GridDim>::X) noexcept {
  return __ockl_get_global_size(0);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::Y,
                        __HIP_Coordinates<__HIP_BlockDim>::Y) noexcept {
  return __ockl_get_global_size(1);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::Y,
                        __HIP_Coordinates<__HIP_GridDim>::Y) noexcept {
  return __ockl_get_global_size(1);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::Z,
                        __HIP_Coordinates<__HIP_BlockDim>::Z) noexcept {
  return __ockl_get_global_size(2);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::Z,
                        __HIP_Coordinates<__HIP_GridDim>::Z) noexcept {
  return __ockl_get_global_size(2);
}

static constexpr __HIP_Coordinates<__HIP_BlockDim> blockDim{};
static constexpr __HIP_Coordinates<__HIP_BlockIdx> blockIdx{};
static constexpr __HIP_Coordinates<__HIP_GridDim> gridDim{};
static constexpr __HIP_Coordinates<__HIP_ThreadIdx> threadIdx{};

extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_id(uint);
#define hipThreadIdx_x (__ockl_get_local_id(0))
#define hipThreadIdx_y (__ockl_get_local_id(1))
#define hipThreadIdx_z (__ockl_get_local_id(2))

extern "C" __device__ __attribute__((const)) size_t __ockl_get_group_id(uint);
#define hipBlockIdx_x (__ockl_get_group_id(0))
#define hipBlockIdx_y (__ockl_get_group_id(1))
#define hipBlockIdx_z (__ockl_get_group_id(2))

extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_size(uint);
#define hipBlockDim_x (__ockl_get_local_size(0))
#define hipBlockDim_y (__ockl_get_local_size(1))
#define hipBlockDim_z (__ockl_get_local_size(2))

extern "C" __device__ __attribute__((const)) size_t __ockl_get_num_groups(uint);
#define hipGridDim_x (__ockl_get_num_groups(0))
#define hipGridDim_y (__ockl_get_num_groups(1))
#define hipGridDim_z (__ockl_get_num_groups(2))

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

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
// Support std::complex.
#if !_OPENMP || __HIP_ENABLE_CUDA_WRAPPER_FOR_OPENMP__
#pragma push_macro("__CUDA__")
#define __CUDA__
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
// Workaround for using libc++ with HIP-Clang.
// The following headers requires clang include path before standard C++ include path.
// However libc++ include path requires to be before clang include path.
// To workaround this, we pass -isystem with the parent directory of clang include
// path instead of the clang include path itself.
#include <include/cuda_wrappers/algorithm>
#include <include/cuda_wrappers/complex>
#include <include/cuda_wrappers/new>
#undef __CUDA__
#pragma pop_macro("__CUDA__")
#endif // !_OPENMP || __HIP_ENABLE_CUDA_WRAPPER_FOR_OPENMP__
#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#endif // defined(__clang__) && defined(__HIP__)

#include <hip/hcc_detail/hip_memory.h>

#endif  // HIP_HCC_DETAIL_RUNTIME_H
