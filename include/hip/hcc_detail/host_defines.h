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
 *  @file  hcc_detail/host_defines.h
 *  @brief TODO-doc
 */

#ifndef HOST_DEFINES_H
#define HOST_DEFINES_H

#ifdef __HCC__
/**
 * Function and kernel markers
 */
#define __host__     __attribute__((cpu))
#define __device__   __attribute__((hc))

#define __global__  __attribute__((hc_grid_launch))

#define __noinline__      __attribute__((noinline))
#define __forceinline__   __attribute__((always_inline))



/*
 * Variable Type Qualifiers:
 */
// _restrict is supported by the compiler
#define __shared__     tile_static
#define __constant__   __attribute__((address_space(1)))

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
