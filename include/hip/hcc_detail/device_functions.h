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

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_DEVICE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_DEVICE_FUNCTIONS_H

#include "host_defines.h"
#include "math_fwd.h"

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <hip/hcc_detail/device_library_decls.h>
#include <hip/hcc_detail/llvm_intrinsics.h>

extern "C" __device__ unsigned int __hip_hc_ir_umul24_int(unsigned int, unsigned int);
extern "C" __device__ signed int __hip_hc_ir_mul24_int(signed int, signed int);
extern "C" __device__ signed int __hip_hc_ir_mulhi_int(signed int, signed int);
extern "C" __device__ unsigned int __hip_hc_ir_umulhi_int(unsigned int, unsigned int);
extern "C" __device__ unsigned int __hip_hc_ir_usad_int(unsigned int, unsigned int, unsigned int);

// integer intrinsic function __poc __clz __ffs __brev
__device__ unsigned int __brev(unsigned int x);
__device__ unsigned long long int __brevll(unsigned long long int x);
__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
__device__ unsigned int __clz(int x);
__device__ unsigned int __clzll(long long int x);
__device__ unsigned int __ffs(int x);
__device__ unsigned int __ffsll(long long int x);
__device__ static unsigned int __hadd(int x, int y);
__device__ static int __mul24(int x, int y);
__device__ long long int __mul64hi(long long int x, long long int y);
__device__ static int __mulhi(int x, int y);
__device__ unsigned int __popc(unsigned int x);
__device__ unsigned int __popcll(unsigned long long int x);
__device__ static int __rhadd(int x, int y);
__device__ static unsigned int __sad(int x, int y, int z);
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);
__device__ static int __umul24(unsigned int x, unsigned int y);
__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

__device__ static inline unsigned int __hadd(int x, int y) {
    int z = x + y;
    int sign = z & 0x8000000;
    int value = z & 0x7FFFFFFF;
    return ((value) >> 1 || sign);
}
__device__ static inline int __mul24(int x, int y) { return __hip_hc_ir_mul24_int(x, y); }
__device__ static inline int __mulhi(int x, int y) { return __hip_hc_ir_mulhi_int(x, y); }
__device__ static inline int __rhadd(int x, int y) {
    int z = x + y + 1;
    int sign = z & 0x8000000;
    int value = z & 0x7FFFFFFF;
    return ((value) >> 1 || sign);
}
__device__ static inline unsigned int __sad(int x, int y, int z) {
    return x > y ? x - y + z : y - x + z;
}
__device__ static inline unsigned int __uhadd(unsigned int x, unsigned int y) {
    return (x + y) >> 1;
}
__device__ static inline int __umul24(unsigned int x, unsigned int y) {
    return __hip_hc_ir_umul24_int(x, y);
}
__device__ static inline unsigned int __umulhi(unsigned int x, unsigned int y) {
    return __hip_hc_ir_umulhi_int(x, y);
}
__device__ static inline unsigned int __urhadd(unsigned int x, unsigned int y) {
    return (x + y + 1) >> 1;
}
__device__ static inline unsigned int __usad(unsigned int x, unsigned int y, unsigned int z) {
    return __hip_hc_ir_usad_int(x, y, z);
}

extern __device__ __attribute__((const)) unsigned int __mbcnt_lo(unsigned int x, unsigned int y) __asm("llvm.amdgcn.mbcnt.lo");
extern __device__ __attribute__((const)) unsigned int __mbcnt_hi(unsigned int x, unsigned int y) __asm("llvm.amdgcn.mbcnt.hi");

__device__ static inline unsigned int __lane_id() { return  __mbcnt_hi(-1, __mbcnt_lo(-1, 0)); }

/*
Rounding modes are not yet supported in HIP
*/

__device__ float __double2float_rd(double x);
__device__ float __double2float_rn(double x);
__device__ float __double2float_ru(double x);
__device__ float __double2float_rz(double x);

__device__ int __double2hiint(double x);

__device__ int __double2int_rd(double x);
__device__ int __double2int_rn(double x);
__device__ int __double2int_ru(double x);
__device__ int __double2int_rz(double x);

__device__ long long int __double2ll_rd(double x);
__device__ long long int __double2ll_rn(double x);
__device__ long long int __double2ll_ru(double x);
__device__ long long int __double2ll_rz(double x);

__device__ int __double2loint(double x);

__device__ unsigned int __double2uint_rd(double x);
__device__ unsigned int __double2uint_rn(double x);
__device__ unsigned int __double2uint_ru(double x);
__device__ unsigned int __double2uint_rz(double x);

__device__ unsigned long long int __double2ull_rd(double x);
__device__ unsigned long long int __double2ull_rn(double x);
__device__ unsigned long long int __double2ull_ru(double x);
__device__ unsigned long long int __double2ull_rz(double x);

__device__ long long int __double_as_longlong(double x);
/*
__device__ unsigned short __float2half_rn(float x);
__device__ float __half2float(unsigned short);

The above device function are not a valid .
Use
__device__ __half __float2half_rn(float x);
__device__ float __half2float(__half);
from hip_fp16.h

CUDA implements half as unsigned short whereas, HIP doesn't.

*/

__device__ int __float2int_rd(float x);
__device__ int __float2int_rn(float x);
__device__ int __float2int_ru(float x);
__device__ int __float2int_rz(float x);

__device__ long long int __float2ll_rd(float x);
__device__ long long int __float2ll_rn(float x);
__device__ long long int __float2ll_ru(float x);
__device__ long long int __float2ll_rz(float x);

__device__ unsigned int __float2uint_rd(float x);
__device__ unsigned int __float2uint_rn(float x);
__device__ unsigned int __float2uint_ru(float x);
__device__ unsigned int __float2uint_rz(float x);

__device__ unsigned long long int __float2ull_rd(float x);
__device__ unsigned long long int __float2ull_rn(float x);
__device__ unsigned long long int __float2ull_ru(float x);
__device__ unsigned long long int __float2ull_rz(float x);

__device__ int __float_as_int(float x);
__device__ unsigned int __float_as_uint(float x);
__device__ double __hiloint2double(int hi, int lo);
__device__ double __int2double_rn(int x);

__device__ float __int2float_rd(int x);
__device__ float __int2float_rn(int x);
__device__ float __int2float_ru(int x);
__device__ float __int2float_rz(int x);

__device__ float __int_as_float(int x);

__device__ double __ll2double_rd(long long int x);
__device__ double __ll2double_rn(long long int x);
__device__ double __ll2double_ru(long long int x);
__device__ double __ll2double_rz(long long int x);

__device__ float __ll2float_rd(long long int x);
__device__ float __ll2float_rn(long long int x);
__device__ float __ll2float_ru(long long int x);
__device__ float __ll2float_rz(long long int x);

__device__ double __longlong_as_double(long long int x);

__device__ double __uint2double_rn(int x);

__device__ float __uint2float_rd(unsigned int x);
__device__ float __uint2float_rn(unsigned int x);
__device__ float __uint2float_ru(unsigned int x);
__device__ float __uint2float_rz(unsigned int x);

__device__ float __uint_as_float(unsigned int x);

__device__ double __ull2double_rd(unsigned long long int x);
__device__ double __ull2double_rn(unsigned long long int x);
__device__ double __ull2double_ru(unsigned long long int x);
__device__ double __ull2double_rz(unsigned long long int x);

__device__ float __ull2float_rd(unsigned long long int x);
__device__ float __ull2float_rn(unsigned long long int x);
__device__ float __ull2float_ru(unsigned long long int x);
__device__ float __ull2float_rz(unsigned long long int x);

__device__ char4 __hip_hc_add8pk(char4, char4);
__device__ char4 __hip_hc_sub8pk(char4, char4);
__device__ char4 __hip_hc_mul8pk(char4, char4);

#if defined(__HCC__)
#define __HCC_OR_HIP_CLANG__ 1
#elif defined(__clang__) && defined(__HIP__)
#define __HCC_OR_HIP_CLANG__ 1
#else
#define __HCC_OR_HIP_CLANG__ 0
#endif

#ifdef __HCC_OR_HIP_CLANG__

#ifdef __HIP_DEVICE_COMPILE__

// Clock functions
__device__
inline
long long int __clock64() { return (long long int) __builtin_amdgcn_s_memrealtime(); }

__device__
inline
long long int  __clock() { return (long long int) __builtin_amdgcn_s_memrealtime(); }

// hip.amdgcn.bc - named sync
__device__
inline
void __named_sync(int a, int b) { __builtin_amdgcn_s_barrier(); }

#endif // __HIP_DEVICE_COMPILE__

// warp vote function __all __any __ballot
__device__
int __all(int input);
__device__
int __any(int input);
__device__
unsigned long long int __ballot(int input);

__device__
inline
uint64_t __ballot64(int a) {
    int64_t s;
    // define i64 @__ballot64(i32 %a) #0 {
    //   %b = tail call i64 asm "v_cmp_ne_i32_e64 $0, 0, $1", "=s,v"(i32 %a) #1
    //   ret i64 %b
    // }
    __asm("v_cmp_ne_i32_e64 $0, 0, $1" : "=s"(s) : "v"(a));
    return s;
}

// hip.amdgcn.bc - lanemask
__device__
inline
int64_t  __lanemask_gt()
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

__device__
inline
void* __get_dynamicgroupbaseptr()
{
    // Get group segment base pointer.
    return (char*)__local_to_generic(__to_local(__llvm_amdgcn_groupstaticsize()));
}

__device__
inline
void *__amdgcn_get_dynamicgroupbaseptr() {
    return __get_dynamicgroupbaseptr();
}

#endif // __HCC_OR_HIP_CLANG__

#ifdef __HCC__

/**
 * extern __shared__
 */

// Macro to replace extern __shared__ declarations
// to local variable definitions
#define HIP_DYNAMIC_SHARED(type, var) type* var = (type*)__get_dynamicgroupbaseptr();

#define HIP_DYNAMIC_SHARED_ATTRIBUTE


#elif defined(__clang__) && defined(__HIP__)

#pragma push_macro("__DEVICE__")
#define __DEVICE__ extern "C" __device__ __attribute__((always_inline)) \
  __attribute__((weak))

__DEVICE__
inline
void __assert_fail(const char * __assertion,
                                     const char *__file,
                                     unsigned int __line,
                                     const char *__function)
{
    // Ignore all the args for now.
    __builtin_trap();
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
    __builtin_trap();
}

// hip.amdgcn.bc - sync threads
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

__device__
inline
static void __work_group_barrier(__cl_mem_fence_flags flags, __memory_scope scope)
{
    if (flags) {
        switch(scope) {
          case __memory_scope_work_item:        break;
          case __memory_scope_sub_group:        __llvm_fence_rel_sg(); break;
          case __memory_scope_work_group:       __llvm_fence_rel_wg(); break;
          case __memory_scope_device:           __llvm_fence_rel_dev(); break;
          case __memory_scope_all_svm_devices:  __llvm_fence_rel_sys(); break;
        }
        //atomic_work_item_fence(flags, memory_order_release, scope);
        __builtin_amdgcn_s_barrier();
        //atomic_work_item_fence(flags, memory_order_acquire, scope);
        switch(scope) {
          case __memory_scope_work_item:        break;
          case __memory_scope_sub_group:        __llvm_fence_acq_sg(); break;
          case __memory_scope_work_group:       __llvm_fence_acq_wg(); break;
          case __memory_scope_device:           __llvm_fence_acq_dev(); break;
          case __memory_scope_all_svm_devices:  __llvm_fence_acq_sys(); break;
        }
    } else {
        __builtin_amdgcn_s_barrier();
    }
}

__device__
inline
static void __barrier(int n)
{
  __work_group_barrier((__cl_mem_fence_flags)n, __memory_scope_work_group);
}

__device__
inline
__attribute__((noduplicate))
void __syncthreads()
{
  __barrier(__CLK_LOCAL_MEM_FENCE);
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
    unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE, HW_ID_CU_ID_OFFSET, HW_ID));
    unsigned se_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SE_ID_SIZE, HW_ID_SE_ID_OFFSET, HW_ID));

    /* Each shader engine has 16 CU */
    return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
}

#pragma push_macro("__DEVICE__")

// Macro to replace extern __shared__ declarations
// to local variable definitions
#define HIP_DYNAMIC_SHARED(type, var) \
    type* var = (type*)__amdgcn_get_dynamicgroupbaseptr();

#define HIP_DYNAMIC_SHARED_ATTRIBUTE


#endif //defined(__clang__) && defined(__HIP__)

#endif
