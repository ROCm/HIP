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
 *  @file  amd_detail/device_library_decls.h
 *  @brief Contains declarations for types and functions in device library.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H

#include "hip/amd_detail/host_defines.h"

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ullong;

extern "C" __device__ __attribute__((const)) bool __ockl_wfany_i32(int);
extern "C" __device__ __attribute__((const)) bool __ockl_wfall_i32(int);
extern "C" __device__ uint __ockl_activelane_u32(void);

extern "C" __device__ __attribute__((const)) uint __ockl_mul24_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul24_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_mul_hi_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul_hi_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_sadd_u32(uint, uint, uint);

extern "C" __device__ __attribute__((const)) uchar __ockl_clz_u8(uchar);
extern "C" __device__ __attribute__((const)) ushort __ockl_clz_u16(ushort);
extern "C" __device__ __attribute__((const)) uint __ockl_clz_u32(uint);
extern "C" __device__ __attribute__((const)) ullong __ockl_clz_u64(ullong);

extern "C" __device__ __attribute__((const)) float __ocml_floor_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_rint_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_ceil_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_trunc_f32(float);

extern "C" __device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
extern "C" __device__ __attribute__((const)) float __ocml_fmax_f32(float, float);

extern "C" __device__ __attribute__((convergent)) void __ockl_gws_init(uint nwm1, uint rid);
extern "C" __device__ __attribute__((convergent)) void __ockl_gws_barrier(uint nwm1, uint rid);

extern "C" __device__ __attribute__((const)) uint32_t __ockl_lane_u32();
extern "C" __device__ __attribute__((const)) int __ockl_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_grid_sync(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_num_grids(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_grid_rank(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_size(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_thread_rank(void);
extern "C" __device__ __attribute__((const)) int __ockl_multi_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_multi_grid_sync(void);

extern "C" __device__ void __ockl_atomic_add_noret_f32(float*, float);

extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_add_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_and_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_or_i32(int a);


// Introduce local address space
#define __local __attribute__((address_space(3)))

#ifdef __HIP_DEVICE_COMPILE__
__device__ inline static __local void* __to_local(unsigned x) { return (__local void*)x; }
#endif //__HIP_DEVICE_COMPILE__

// Using hip.amdgcn.bc - sync threads
#define __CLK_LOCAL_MEM_FENCE    0x01
typedef unsigned __cl_mem_fence_flags;

typedef enum __memory_scope {
  __memory_scope_work_item = __OPENCL_MEMORY_SCOPE_WORK_ITEM,
  __memory_scope_work_group = __OPENCL_MEMORY_SCOPE_WORK_GROUP,
  __memory_scope_device = __OPENCL_MEMORY_SCOPE_DEVICE,
  __memory_scope_all_svm_devices = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
  __memory_scope_sub_group = __OPENCL_MEMORY_SCOPE_SUB_GROUP
} __memory_scope;

// enum values aligned with what clang uses in EmitAtomicExpr()
typedef enum __memory_order
{
  __memory_order_relaxed = __ATOMIC_RELAXED,
  __memory_order_acquire = __ATOMIC_ACQUIRE,
  __memory_order_release = __ATOMIC_RELEASE,
  __memory_order_acq_rel = __ATOMIC_ACQ_REL,
  __memory_order_seq_cst = __ATOMIC_SEQ_CST
} __memory_order;

// Linked from hip.amdgcn.bc
extern "C" __device__ void
__atomic_work_item_fence(__cl_mem_fence_flags, __memory_order, __memory_scope);

#endif
