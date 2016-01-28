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
 * @file hip_runtime_api.h
 *
 * @brief Defines the API signatures for HIP runtime.
 * This file can be compiled with a standard compiler.
 */

#pragma once


#include <string.h> // for getDeviceProp
#include <hip_common.h>

typedef struct {
    // 32-bit Atomics:
    unsigned hasGlobalInt32Atomics    : 1;   ///< 32-bit integer atomics for global memory
    unsigned hasGlobalFloatAtomicExch : 1;   ///< 32-bit float atomic exch for global memory
    unsigned hasSharedInt32Atomics    : 1;   ///< 32-bit integer atomics for shared memory
    unsigned hasSharedFloatAtomicExch : 1;   ///< 32-bit float atomic exch for shared memory
    unsigned hasFloatAtomicAdd        : 1;   ///< 32-bit float atomic add in global and shared memory

    // 64-bit Atomics:
    unsigned hasGlobalInt64Atomics    : 1;   ///< 64-bit integer atomics for global memory
    unsigned hasSharedInt64Atomics    : 1;   ///< 64-bit integer atomics for shared memory

    // Doubles
    unsigned hasDoubles               : 1;   ///< double-precision floating point.

    // Warp cross-lane operations:
    unsigned hasWarpVote              : 1;   ///< warp vote instructions (__any, __all)
    unsigned hasWarpBallot            : 1;   ///< warp ballot instructions (__ballot)
    unsigned hasWarpShuffle           : 1;   ///< warp shuffle operations. (__shfl_*)
    unsigned hasFunnelShift           : 1;   ///< funnel two words into one, with shift&mask caps

    // Sync
    unsigned hasThreadFenceSystem     : 1;   ///< __threadfence_system
    unsigned hasSyncThreadsExt        : 1;   ///< __syncthreads_count, syncthreads_and, syncthreads_or

    // Misc
    unsigned hasSurfaceFuncs          : 1;   ///< Surface functions
    unsigned has3dGrid                : 1;   ///< Grid and group dims are 3D (rather than 2D)
    unsigned hasDynamicParallelism    : 1;   ///< Dynamic parallelism
} hipDeviceArch_t;


//---
// Common headers for both NVCC and HCC paths:

/**
 * hipDeviceProp
 *
 */
typedef struct hipDeviceProp_t {
	char name[256];           ///< Device name
	size_t totalGlobalMem;    ///< Size of global memory region (in bytes)
	size_t sharedMemPerBlock; ///< Size of shared memory region (in bytes)
	int regsPerBlock ; ///< registers per block
	int warpSize ; ///< warp size
	int maxThreadsPerBlock; ///< max work items per work group or workgroup max size
	int maxThreadsDim[3]; ///< max number of threads in each dimension (XYZ) of a block
	int maxGridSize[3]; ///< max grid dimensions (XYZ)
	int clockRate ; ///< max clock frequency of the multiProcessors, in khz.

	size_t totalConstMem; ///< Size of shared memory region (in bytes)
	int major ; ///< Major compute capability.  On HCC, this is an approximation and features may differ from CUDA CC.  See the arch feature flags for portable ways to query feature caps.
	int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may differ from CUDA CC.  See the arch feature flags for portable ways to query feature caps.
	int multiProcessorCount; ///< number of multi-processors (compute units)
	int l2CacheSize; ///< L2 cache size
	int maxThreadsPerMultiProcessor; ///< Maximum resident threads per multi-processor
	int computeMode; ///< Compute mode

	int clockInstructionRate ;   ///< Frequency in khz of the timer used by the device-side "clock*" instructions.  New for HIP.

    hipDeviceArch_t arch;  ///< Architectural feature flags.  New for HIP.
 } hipDeviceProp_t;


// hack to get these to show up in Doxygen:
/**
 *     @defgroup GlobalDefs Global enum and defines
 *     @{
 *
 */


/*
 * @brief hipError_t
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipError_t {
   hipSuccess = 0                  ///< Successful completion.
  ,hipErrorMemoryAllocation        ///< Memory allocation error.
  ,hipErrorMemoryFree              ///< Memory free error.
  ,hipErrorUnknownSymbol           ///< Unknown symbol
  ,hipErrorOutOfResources          ///< Out of resources error
  ,hipErrorInvalidValue            ///< One or more of the parameters passed to the API call is NULL or not in an acceptable range.
  ,hipErrorInvalidResourceHandle   ///< Resource handle (hipEvent_t or hipStream_t) invalid.
  ,hipErrorInvalidDevice           ///< DeviceID must be in range 0...#compute-devices.
  ,hipErrorNoDevice                ///< Call to hipGetDeviceCount returned 0 devices
  ,hipErrorNotReady                ///< indicates that asynchronous operations enqueued earlier are not ready.  This is not actually an error, but is used to distinguish from hipSuccess (which indicates completion).  APIs that return this error include hipEventQuery and hipStreamQuery.

  ,hipErrorUnknown                 ///< Unknown error
  ,hipErrorTbd                     ///< Marker that more error codes are needed.
} hipError_t;



/**
 *     @}
 */

#if defined(__HIP_PLATFORM_HCC__) and not defined (__HIP_PLATFORM_NVCC__)
#include "hcc_detail/hip_runtime_api.h"
#elif defined(__HIP_PLATFORM_NVCC__) and not defined (__HIP_PLATFORM_HCC__)
#include "nvcc_detail/hip_runtime_api.h"
#else
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif


/**
 * @brief: C++ wrapper for hipMalloc
 *
 * Perform automatic type conversion to eliminate need for excessive typecasting (ie void**)
 *
 * @see hipMalloc
 */
#ifdef __cplusplus
template<class T>
static inline hipError_t hipMalloc ( T** devPtr, size_t size)
{
    return hipMalloc((void**)devPtr, size);
}

template<class T>
static inline hipError_t hipMallocHost ( T** ptr, size_t size)
{
    return hipMallocHost((void**)ptr, size);
}
#endif
