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
 *  @file  hcc_detail/device_library_decls.h
 *  @brief Contains declarations for types and functions in device library.
 */

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_DEVICE_LIBRARY_DECLS_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_DEVICE_LIBRARY_DECLS_H

#include "hip/hcc_detail/host_defines.h"

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

extern "C" __device__ __attribute__((const)) bool __ockl_wfany_i32(int);
extern "C" __device__ __attribute__((const)) bool __ockl_wfall_i32(int);
extern "C" __device__ uint __ockl_activelane_u32(void);

extern "C" __device__ __attribute__((const)) uint __ockl_mul24_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul24_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_mul_hi_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul_hi_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_sad_u32(uint, uint, uint);

extern "C" __device__ __attribute__((const)) uchar __ockl_clz_u8(uchar);
extern "C" __device__ __attribute__((const)) ushort __ockl_clz_u16(ushort);
extern "C" __device__ __attribute__((const)) uint __ockl_clz_u32(uint);
extern "C" __device__ __attribute__((const)) ulong __ockl_clz_u64(ulong);

extern "C" __device__ __attribute__((const)) float __ocml_floor_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_rint_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_ceil_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_trunc_f32(float);

extern "C" __device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
extern "C" __device__ __attribute__((const)) float __ocml_fmax_f32(float, float);

// Introduce local address space
#define __local __attribute__((address_space(3)))

#ifdef __HIP_DEVICE_COMPILE__
__device__ inline static __local void* __to_local(unsigned x) { return (__local void*)x; }
#endif //__HIP_DEVICE_COMPILE__

// __llvm_fence* functions from device-libs/irif/src/fence.ll
extern "C" __device__ void __llvm_fence_acq_sg(void);
extern "C" __device__ void __llvm_fence_acq_wg(void);
extern "C" __device__ void __llvm_fence_acq_dev(void);
extern "C" __device__ void __llvm_fence_acq_sys(void);

extern "C" __device__ void __llvm_fence_rel_sg(void);
extern "C" __device__ void __llvm_fence_rel_wg(void);
extern "C" __device__ void __llvm_fence_rel_dev(void);
extern "C" __device__ void __llvm_fence_rel_sys(void);

extern "C" __device__ void __llvm_fence_ar_sg(void);
extern "C" __device__ void __llvm_fence_ar_wg(void);
extern "C" __device__ void __llvm_fence_ar_dev(void);
extern "C" __device__ void __llvm_fence_ar_sys(void);


extern "C" __device__ void __llvm_fence_sc_sg(void);
extern "C" __device__ void __llvm_fence_sc_wg(void);
extern "C" __device__ void __llvm_fence_sc_dev(void);
extern "C" __device__ void __llvm_fence_sc_sys(void);

#endif
