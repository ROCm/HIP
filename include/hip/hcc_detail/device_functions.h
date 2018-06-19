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

typedef unsigned long ulong;
typedef unsigned int uint;

extern "C" __device__ inline uint __hip_hc_ir_umul24_int(uint a, uint b) {
    // define i32 @__hip_hc_ir_umul24_int(i32 %a, i32 %b) #1 {
    //   %1 = tail call i32 asm sideeffect "v_mul_u32_u24 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
    //   ret i32 %1
    // }
    uint out;
    __asm volatile("v_mul_u32_u24 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

extern "C" __device__ inline int __hip_hc_ir_mul24_int(int a, int b) {
    // define i32 @__hip_hc_ir_mul24_int(i32 %a, i32 %b) #1 {
    //   %1 = tail call i32 asm sideeffect "v_mul_i32_i24 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
    //   ret i32 %1
    // }
    int out;
    __asm volatile("v_mul_i32_i24 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

extern "C" __device__ inline int __hip_hc_ir_mulhi_int(int a, int b) {
    // define i32 @__hip_hc_ir_mulhi_int(i32 %a, i32 %b) #1 {
    //   %1 = tail call i32 asm sideeffect "v_mul_hi_i32 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
    //   ret i32 %1
    // }
    int out;
    __asm volatile("v_mul_hi_i32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

extern "C" __device__ inline uint __hip_hc_ir_umulhi_int(uint a, uint b) {
    // define i32 @__hip_hc_ir_umulhi_int(i32 %a, i32 %b) #1 {
    //   %1 = tail call i32 asm sideeffect "v_mul_hi_u32 $0, $1, $2","=v,v,v"(i32 %a, i32 %b)
    //   ret i32 %1
    // }
    uint out;
    __asm volatile("v_mul_hi_u32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

extern "C" __device__ inline uint __hip_hc_ir_usad_int(uint a, uint b, uint c) {
    // define i32 @__hip_hc_ir_usad_int(i32 %a, i32 %b, i32 %c) #1 {
    //  %1 = tail call i32 asm sideeffect "v_sad_u32 $0, $1, $2, $3","=v,v,v,v"(i32 %a, i32 %b, i32 %c)
    //  ret i32 %1
    // }
    uint out;
    __asm volatile("v_sad_u32 %0, %1, %2, %3" : "=v"(out) : "v"(a), "v"(b), "v"(c));
    return out;
}

/*
Integer Intrinsics
*/

// integer intrinsic function __poc __clz __ffs __brev
__device__ static inline unsigned int __popc(unsigned int input) {
    return __builtin_popcount(input);
}
__device__ static inline unsigned int __popcll(unsigned long long int input) {
    return __builtin_popcountl(input);
}

__device__ static inline unsigned int __clz(unsigned int input) {
#ifdef NVCC_COMPAT
    return input == 0 ? 32 : __builtin_clz(input);
#else
    return input == 0 ? -1 : __builtin_clz(input);
#endif
}

__device__ static inline unsigned int __clzll(unsigned long long int input) {
#ifdef NVCC_COMPAT
    return input == 0 ? 64 : ( input == 0 ? -1 : __builtin_clzl(input) );
#else
    return input == 0 ? -1 : __builtin_clzl(input);
#endif
}

__device__ static inline unsigned int __clz(int input) {
#ifdef NVCC_COMPAT
    return input == 0 ? 32 : ( input > 0 ? __builtin_clz(input) : __builtin_clz(~input) );
#else
    if (input == 0) return -1;
    return input > 0 ? __builtin_clz(input) : __builtin_clz(~input);
#endif
}

__device__ static inline unsigned int __clzll(long long int input) {
#ifdef NVCC_COMPAT
    return input == 0 ? 64 : input > 0 ? __builtin_clzl(input) : __builtin_clzl(~input);
#else
    if (input == 0) return -1;
    return input > 0 ? __builtin_clzl(input) : __builtin_clzl(~input);
#endif
}

__device__ static inline unsigned int __ffs(unsigned int input) {
#ifdef NVCC_COMPAT
    return ( input == 0 ? -1 : __builtin_ctz(input) ) + 1;
#else
    return input == 0 ? -1 : __builtin_ctz(input);
#endif
}

__device__ static inline unsigned int __ffsll(unsigned long long int input) {
#ifdef NVCC_COMPAT
    return ( input == 0 ? -1 : __builtin_ctzl(input) ) + 1;
#else
    return input == 0 ? -1 : __builtin_ctzl(input);
#endif
}

__device__ static inline unsigned int __ffs(int input) {
#ifdef NVCC_COMPAT
    return ( input == 0 ? -1 : __builtin_ctz(input) ) + 1;
#else
    return input == 0 ? -1 : __builtin_ctz(input);
#endif
}

__device__ static inline unsigned int __ffsll(long long int input) {
#ifdef NVCC_COMPAT
    return ( input == 0 ? -1 : __builtin_ctzl(input) ) + 1;
#else
    return input == 0 ? -1 : __builtin_ctzl(input);
#endif
}

__device__ static inline unsigned int __brev(unsigned int input) { return __llvm_bitrev_b32(input); }

__device__ static inline unsigned long long int __brevll(unsigned long long int input) {
    return __llvm_bitrev_b64(input);
}

__device__ static unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
__device__ static unsigned int __hadd(int x, int y);
__device__ static int __mul24(int x, int y);
__device__ static long long int __mul64hi(long long int x, long long int y);
__device__ static int __mulhi(int x, int y);
__device__ static int __rhadd(int x, int y);
__device__ static unsigned int __sad(int x, int y, int z);
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);
__device__ static int __umul24(unsigned int x, unsigned int y);
__device__ static unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

struct ucharHolder {
    union {
        unsigned char c[4];
        unsigned int ui;
    };
} __attribute__((aligned(4)));

struct uchar2Holder {
    union {
        unsigned int ui[2];
        unsigned char c[8];
    };
} __attribute__((aligned(8)));

__device__
static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    struct uchar2Holder cHoldVal;
    struct ucharHolder cHoldKey;
    struct ucharHolder cHoldOut;
    cHoldKey.ui = s;
    cHoldVal.ui[0] = x;
    cHoldVal.ui[1] = y;
    cHoldOut.c[0] = cHoldVal.c[cHoldKey.c[0]];
    cHoldOut.c[1] = cHoldVal.c[cHoldKey.c[1]];
    cHoldOut.c[2] = cHoldVal.c[cHoldKey.c[2]];
    cHoldOut.c[3] = cHoldVal.c[cHoldKey.c[3]];
    return cHoldOut.ui;
}

__device__ static inline unsigned int __hadd(int x, int y) {
    int z = x + y;
    int sign = z & 0x8000000;
    int value = z & 0x7FFFFFFF;
    return ((value) >> 1 || sign);
}
__device__ static inline int __mul24(int x, int y) { return __hip_hc_ir_mul24_int(x, y); }

__device__ static inline long long __mul64hi(long long int x, long long int y) {
    ulong x0 = (ulong)x & 0xffffffffUL;
    long x1 = x >> 32;
    ulong y0 = (ulong)y & 0xffffffffUL;
    long y1 = y >> 32;
    ulong z0 = x0*y0;
    long t = x1*y0 + (z0 >> 32);
    long z1 = t & 0xffffffffL;
    long z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

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

__device__
static inline unsigned long long __umul64hi(unsigned long long int x, unsigned long long int y) {
    ulong x0 = x & 0xffffffffUL;
    ulong x1 = x >> 32;
    ulong y0 = y & 0xffffffffUL;
    ulong y1 = y >> 32;
    ulong z0 = x0*y0;
    ulong t = x1*y0 + (z0 >> 32);
    ulong z1 = t & 0xffffffffUL;
    ulong z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
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

__device__ static inline unsigned int __lane_id() { return  __mbcnt_hi(-1, __mbcnt_lo(-1, 0)); }

/*
HIP specific device functions
*/

// utility union type
union __u {
    int i;
    unsigned int u;
    float f;
};

__device__ static inline unsigned __hip_ds_bpermute(int index, unsigned src) {
    __u tmp; tmp.u = src;
    tmp.i = __llvm_amdgcn_ds_bpermute(index, tmp.i);
    return tmp.u;
}

__device__ static inline float __hip_ds_bpermutef(int index, float src) {
    __u tmp; tmp.f = src;
    tmp.i = __llvm_amdgcn_ds_bpermute(index, tmp.i);
    return tmp.f;
}

__device__ static inline unsigned __hip_ds_permute(int index, unsigned src) {
  __u tmp; tmp.u = src;
  tmp.i = __llvm_amdgcn_ds_permute(index, tmp.i);
  return tmp.u;
}

__device__ static inline float __hip_ds_permutef(int index, float src) {
  __u tmp; tmp.u = src;
  tmp.i = __llvm_amdgcn_ds_permute(index, tmp.i);
  return tmp.u;
}

__device__ static inline unsigned __hip_ds_swizzle(unsigned int src, int pattern) {
    __u tmp; tmp.u = src;
    tmp.i = __llvm_amdgcn_ds_swizzle(tmp.i, pattern);
    return tmp.u;
}
__device__ static inline float __hip_ds_swizzlef(float src, int pattern) {
    __u tmp; tmp.f = src;
    tmp.i = __llvm_amdgcn_ds_swizzle(tmp.i, pattern);
    return tmp.f;
}

__device__ static inline int __hip_move_dpp(int src, int dpp_ctrl, int row_mask,
                                            int bank_mask, bool bound_ctrl) {
    return __llvm_amdgcn_move_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}

#define MASK1 0x00ff00ff
#define MASK2 0xff00ff00

__device__ static inline char4 __hip_hc_add8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.a & MASK1;
    unsigned one2 = in2.a & MASK1;
    out.a = (one1 + one2) & MASK1;
    one1 = in1.a & MASK2;
    one2 = in2.a & MASK2;
    out.a = out.a | ((one1 + one2) & MASK2);
    return out;
}

__device__ static inline char4 __hip_hc_sub8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.a & MASK1;
    unsigned one2 = in2.a & MASK1;
    out.a = (one1 - one2) & MASK1;
    one1 = in1.a & MASK2;
    one2 = in2.a & MASK2;
    out.a = out.a | ((one1 - one2) & MASK2);
    return out;
}

__device__ static inline char4 __hip_hc_mul8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.a & MASK1;
    unsigned one2 = in2.a & MASK1;
    out.a = (one1 * one2) & MASK1;
    one1 = in1.a & MASK2;
    one2 = in2.a & MASK2;
    out.a = out.a | ((one1 * one2) & MASK2);
    return out;
}

/*
Rounding modes are not yet supported in HIP
*/

__device__ static inline float __double2float_rd(double x) { return (double)x; }
__device__ static inline float __double2float_rn(double x) { return (double)x; }
__device__ static inline float __double2float_ru(double x) { return (double)x; }
__device__ static inline float __double2float_rz(double x) { return (double)x; }

__device__ static inline int __double2hiint(double x) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");

    int tmp[2];
    __builtin_memcpy(tmp, &x, sizeof(tmp));

    return tmp[1];
}
__device__ static inline int __double2loint(double x) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");

    int tmp[2];
    __builtin_memcpy(tmp, &x, sizeof(tmp));

    return tmp[0];
}

__device__ static inline int __double2int_rd(double x) { return (int)x; }
__device__ static inline int __double2int_rn(double x) { return (int)x; }
__device__ static inline int __double2int_ru(double x) { return (int)x; }
__device__ static inline int __double2int_rz(double x) { return (int)x; }

__device__ static inline long long int __double2ll_rd(double x) { return (long long int)x; }
__device__ static inline long long int __double2ll_rn(double x) { return (long long int)x; }
__device__ static inline long long int __double2ll_ru(double x) { return (long long int)x; }
__device__ static inline long long int __double2ll_rz(double x) { return (long long int)x; }

__device__ static inline unsigned int __double2uint_rd(double x) { return (unsigned int)x; }
__device__ static inline unsigned int __double2uint_rn(double x) { return (unsigned int)x; }
__device__ static inline unsigned int __double2uint_ru(double x) { return (unsigned int)x; }
__device__ static inline unsigned int __double2uint_rz(double x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __double2ull_rd(double x) {
    return (unsigned long long int)x;
}
__device__ static inline unsigned long long int __double2ull_rn(double x) {
    return (unsigned long long int)x;
}
__device__ static inline unsigned long long int __double2ull_ru(double x) {
    return (unsigned long long int)x;
}
__device__ static inline unsigned long long int __double2ull_rz(double x) {
    return (unsigned long long int)x;
}

__device__ static inline long long int __double_as_longlong(double x) {
    static_assert(sizeof(long long) == sizeof(double), "");

    long long tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

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

__device__ static inline int __float2int_rd(float x) { return (int)__ocml_floor_f32(x); }
__device__ static inline int __float2int_rn(float x) { return (int)__ocml_rint_f32(x); }
__device__ static inline int __float2int_ru(float x) { return (int)__ocml_ceil_f32(x); }
__device__ static inline int __float2int_rz(float x) { return (int)__ocml_trunc_f32(x); }

__device__ static inline long long int __float2ll_rd(float x) { return (long long int)x; }
__device__ static inline long long int __float2ll_rn(float x) { return (long long int)x; }
__device__ static inline long long int __float2ll_ru(float x) { return (long long int)x; }
__device__ static inline long long int __float2ll_rz(float x) { return (long long int)x; }

__device__ static inline unsigned int __float2uint_rd(float x) { return (unsigned int)x; }
__device__ static inline unsigned int __float2uint_rn(float x) { return (unsigned int)x; }
__device__ static inline unsigned int __float2uint_ru(float x) { return (unsigned int)x; }
__device__ static inline unsigned int __float2uint_rz(float x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __float2ull_rd(float x) {
    return (unsigned long long int)x;
}
__device__ static inline unsigned long long int __float2ull_rn(float x) {
    return (unsigned long long int)x;
}
__device__ static inline unsigned long long int __float2ull_ru(float x) {
    return (unsigned long long int)x;
}
__device__ static inline unsigned long long int __float2ull_rz(float x) {
    return (unsigned long long int)x;
}

__device__ static inline int __float_as_int(float x) {
    static_assert(sizeof(int) == sizeof(float), "");

    int tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline unsigned int __float_as_uint(float x) {
    static_assert(sizeof(unsigned int) == sizeof(float), "");

    unsigned int tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __hiloint2double(int hi, int lo) {
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    uint64_t tmp0 = (static_cast<uint64_t>(hi) << 32ull) | static_cast<uint32_t>(lo);
    double tmp1;
    __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));

    return tmp1;
}

__device__ static inline double __int2double_rn(int x) { return (double)x; }

__device__ static inline float __int2float_rd(int x) { return (float)x; }
__device__ static inline float __int2float_rn(int x) { return (float)x; }
__device__ static inline float __int2float_ru(int x) { return (float)x; }
__device__ static inline float __int2float_rz(int x) { return (float)x; }

__device__ static inline float __int_as_float(int x) {
    static_assert(sizeof(float) == sizeof(int), "");

    float tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __ll2double_rd(long long int x) { return (double)x; }
__device__ static inline double __ll2double_rn(long long int x) { return (double)x; }
__device__ static inline double __ll2double_ru(long long int x) { return (double)x; }
__device__ static inline double __ll2double_rz(long long int x) { return (double)x; }

__device__ static inline float __ll2float_rd(long long int x) { return (float)x; }
__device__ static inline float __ll2float_rn(long long int x) { return (float)x; }
__device__ static inline float __ll2float_ru(long long int x) { return (float)x; }
__device__ static inline float __ll2float_rz(long long int x) { return (float)x; }

__device__ static inline double __longlong_as_double(long long int x) {
    static_assert(sizeof(double) == sizeof(long long), "");

    double tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return x;
}

__device__ static inline double __uint2double_rn(int x) { return (double)x; }

__device__ static inline float __uint2float_rd(unsigned int x) { return (float)x; }
__device__ static inline float __uint2float_rn(unsigned int x) { return (float)x; }
__device__ static inline float __uint2float_ru(unsigned int x) { return (float)x; }
__device__ static inline float __uint2float_rz(unsigned int x) { return (float)x; }

__device__ static inline float __uint_as_float(unsigned int x) {
   static_assert(sizeof(float) == sizeof(unsigned int), "");

    float tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __ull2double_rd(unsigned long long int x) { return (double)x; }
__device__ static inline double __ull2double_rn(unsigned long long int x) { return (double)x; }
__device__ static inline double __ull2double_ru(unsigned long long int x) { return (double)x; }
__device__ static inline double __ull2double_rz(unsigned long long int x) { return (double)x; }

__device__ static inline float __ull2float_rd(unsigned long long int x) { return (float)x; }
__device__ static inline float __ull2float_rn(unsigned long long int x) { return (float)x; }
__device__ static inline float __ull2float_ru(unsigned long long int x) { return (float)x; }
__device__ static inline float __ull2float_rz(unsigned long long int x) { return (float)x; }

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
extern "C" __device__ inline uint64_t __activelanemask_v4_b64_b1(unsigned int input) {
    uint64_t output;
    // define i64 @__activelanemask_v4_b64_b1(i32 %input) #5 {
    //   %a = tail call i64 asm "v_cmp_ne_i32_e64 $0, 0, $1", "=s,v"(i32 %input) #9
    //   ret i64 %a
    // }
    __asm("v_cmp_ne_i32_e64 %0, 0, %1" : "=s"(output) : "v"(input));
    return output;
}

__device__
inline
unsigned int __activelanecount_u32_b1(unsigned int input) {
    return  __popcll(__activelanemask_v4_b64_b1(input));
}

__device__
inline
int __all(int predicate) {
    return __popcll(__activelanemask_v4_b64_b1(predicate)) == __activelanecount_u32_b1(1);
}

__device__
inline
int __any(int predicate) {
#ifdef NVCC_COMPAT
    if (__popcll(__activelanemask_v4_b64_b1(predicate)) != 0)
        return 1;
    else
        return 0;
#else
    return __popcll(__activelanemask_v4_b64_b1(predicate));
#endif
}

__device__
inline
unsigned long long int __ballot(int predicate) {
    return __activelanemask_v4_b64_b1(predicate);
}

__device__
inline
unsigned long long int __ballot64(int predicate) {
    return __activelanemask_v4_b64_b1(predicate);
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
