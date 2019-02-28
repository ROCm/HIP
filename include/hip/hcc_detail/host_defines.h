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
 *  @file  hcc_detail/host_defines.h
 *  @brief TODO-doc
 */

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HOST_DEFINES_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HOST_DEFINES_H


// Add guard to Generic Grid Launch method
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif

#ifdef __HCC__
/**
 * Function and kernel markers
 */
#define __host__ __attribute__((cpu))
#define __device__ __attribute__((hc))

#if GENERIC_GRID_LAUNCH == 0
#define __global__ __attribute__((hc_grid_launch)) __attribute__((used))
#else
#if __hcc_workweek__ >= 17481
#define __global__ __attribute__((annotate("__HIP_global_function__"), cpu, hc, used))
#else
#define __global__ __attribute__((hc, used))
#endif
#endif  // GENERIC_GRID_LAUNCH

#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline __attribute__((always_inline))


/*
 * Variable Type Qualifiers:
 */
// _restrict is supported by the compiler
#define __shared__ tile_static
#define __constant__ __attribute__((hc))

#elif defined(__clang__) && defined(__HIP__)

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline __attribute__((always_inline))

#else

// Non-HCC compiler
/**
 * Function and kernel markers
 */
#define __host__
#define __device__

#define __global__

#define __noinline__
#define __forceinline__

#define __shared__
#define __constant__

#endif

#endif
