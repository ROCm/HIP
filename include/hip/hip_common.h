/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_COMMON_H
#define HIP_INCLUDE_HIP_HIP_COMMON_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#endif
// Common code included at start of every hip file.
// Auto enable __HIP_PLATFORM_AMD__ if compiling on AMD platform
// Other compiler (GCC,ICC,etc) need to set one of these macros explicitly
#if defined(__clang__) && defined(__HIP__)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#endif  // defined(__clang__) && defined(__HIP__)

// Auto enable __HIP_PLATFORM_NVIDIA__ if compiling with NVIDIA platform
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__) && !defined(__HIP__))
#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_NVIDIA__
#endif

#ifdef __CUDACC__
#define __HIPCC__
#endif

#endif  //__NVCC__

// Auto enable __HIP_DEVICE_COMPILE__ if compiled in HCC or NVCC device path
#if (defined(__HCC_ACCELERATOR__) && __HCC_ACCELERATOR__ != 0) ||                                  \
    (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)
#define __HIP_DEVICE_COMPILE__ 1
#endif

#ifdef __GNUC__
#define HIP_PUBLIC_API __attribute__((visibility("default")))
#define HIP_INTERNAL_EXPORTED_API __attribute__((visibility("default")))
#else
#define HIP_PUBLIC_API
#define HIP_INTERNAL_EXPORTED_API
#endif

#if __HIP_DEVICE_COMPILE__ == 0
// 32-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ (0)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ (0)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ (0)

// 64-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ (0)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ (0)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__ (0)

// Warp cross-lane operations
#define __HIP_ARCH_HAS_WARP_VOTE__ (0)
#define __HIP_ARCH_HAS_WARP_BALLOT__ (0)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__ (0)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ (0)

// Sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ (0)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__ (0)

// Misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__ (0)
#define __HIP_ARCH_HAS_3DGRID__ (0)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ (0)
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
