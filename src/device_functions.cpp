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

#include <hip/device_functions.h>
#include <hc.hpp>
#include <grid_launch.h>
#include <hc_math.hpp>
#include "device_util.h"

struct holder64Bit{
  union{
    double d;
    unsigned long int uli;
    signed long int sli;
    signed int si[2];
    unsigned int ui[2];
  };
} __attribute__((aligned(8)));

struct holder32Bit {
  union {
    float f;
    unsigned int ui;
    signed int si;
  };
} __attribute__((aligned(4)));

__device__ struct holder64Bit hold64;
__device__ struct holder32Bit hold32;

__device__ float __double2float_rd(double x)
{
  return (double)x;
}
__device__ float __double2float_rn(double x)
{
  return (double)x;
}
__device__ float __double2float_ru(double x)
{
  return (double)x;
}
__device__ float __double2float_rz(double x)
{
  return (double)x;
}


__device__ int __double2hiint(double x)
{
  hold64.d = x;
  return hold64.si[1];
}
__device__ int __double2loint(double x)
{
  hold64.d = x;
  return hold64.si[0];
}


__device__ int __double2int_rd(double x)
{
  return (int)x;
}
__device__ int __double2int_rn(double x)
{
  return (int)x;
}
__device__ int __double2int_ru(double x)
{
  return (int)x;
}
__device__ int __double2int_rz(double x)
{
  return (int)x;
}

__device__ long long int __double2ll_rd(double x)
{
  return (long long int)x;
}
__device__ long long int __double2ll_rn(double x)
{
  return (long long int)x;
}
__device__ long long int __double2ll_ru(double x)
{
  return (long long int)x;
}
__device__ long long int __double2ll_rz(double x)
{
  return (long long int)x;
}


__device__ unsigned int __double2uint_rd(double x)
{
  return (unsigned int)x;
}
__device__ unsigned int __double2uint_rn(double x)
{
  return (unsigned int)x;
}
__device__ unsigned int __double2uint_ru(double x)
{
  return (unsigned int)x;
}
__device__ unsigned int __double2uint_rz(double x)
{
  return (unsigned int)x;
}

__device__ unsigned long long int __double2ull_rd(double x)
{
  return (unsigned long long int)x;
}
__device__ unsigned long long int __double2ull_rn(double x)
{
  return (unsigned long long int)x;
}
__device__ unsigned long long int __double2ull_ru(double x)
{
  return (unsigned long long int)x;
}
__device__ unsigned long long int __double2ull_rz(double x)
{
  return (unsigned long long int)x;
}

__device__ long long int __double_as_longlong(double x)
{
  hold64.d = x;
  return hold64.sli;
}

__device__ int __float2int_rd(float x)
{
  return (int)x;
}
__device__ int __float2int_rn(float x)
{
  return (int)x;
}
__device__ int __float2int_ru(float x)
{
  return (int)x;
}
__device__ int __float2int_rz(float x)
{
  return (int)x;
}

__device__ long long int __float2ll_rd(float x)
{
  return (long long int)x;
}
__device__ long long int __float2ll_rn(float x)
{
  return (long long int)x;
}
__device__ long long int __float2ll_ru(float x)
{
  return (long long int)x;
}
__device__ long long int __float2ll_rz(float x)
{
  return (long long int)x;
}

__device__ unsigned int __float2uint_rd(float x)
{
  return (unsigned int)x;
}
__device__ unsigned int __float2uint_rn(float x)
{
  return (unsigned int)x;
}
__device__ unsigned int __float2uint_ru(float x)
{
  return (unsigned int)x;
}
__device__ unsigned int __float2uint_rz(float x)
{
  return (unsigned int)x;
}

__device__ unsigned long long int __float2ull_rd(float x)
{
  return (unsigned long long int)x;
}
__device__ unsigned long long int __float2ull_rn(float x)
{
  return (unsigned long long int)x;
}
__device__ unsigned long long int __float2ull_ru(float x)
{
  return (unsigned long long int)x;
}
__device__ unsigned long long int __float2ull_rz(float x)
{
  return (unsigned long long int)x;
}

__device__ int __float_as_int(float x)
{
  hold32.f = x;
  return hold32.si;
}
__device__ unsigned int __float_as_uint(float x)
{
  hold32.f = x;
  return hold32.ui;
}
__device__ double __hiloint2double(int hi, int lo)
{
  hold64.si[1] = hi;
  hold64.si[0] = lo;
  return hold64.d;
}
__device__ double __int2double_rn(int x)
{
  return (double)x;
}

__device__ float __int2float_rd(int x)
{
  return (float)x;
}
__device__ float __int2float_rn(int x)
{
  return (float)x;
}
__device__ float __int2float_ru(int x)
{
  return (float)x;
}
__device__ float __int2float_rz(int x)
{
  return (float)x;
}

__device__ float __int_as_float(int x)
{
  hold32.si = x;
  return hold32.f;
}

__device__ double __ll2double_rd(long long int x)
{
  return (double)x;
}
__device__ double __ll2double_rn(long long int x)
{
  return (double)x;
}
__device__ double __ll2double_ru(long long int x)
{
  return (double)x;
}
__device__ double __ll2double_rz(long long int x)
{
  return (double)x;
}

__device__ float __ll2float_rd(long long int x)
{
  return (float)x;
}
__device__ float __ll2float_rn(long long int x)
{
  return (float)x;
}
__device__ float __ll2float_ru(long long int x)
{
  return (float)x;
}
__device__ float __ll2float_rz(long long int x)
{
  return (float)x;
}

__device__ double __longlong_as_double(long long int x)
{
  hold64.sli = x;
  return hold64.d;
}

__device__ double __uint2double_rn(int x)
{
  return (double)x;
}

__device__ float __uint2float_rd(unsigned int x)
{
  return (float)x;
}
__device__ float __uint2float_rn(unsigned int x)
{
  return (float)x;
}
__device__ float __uint2float_ru(unsigned int x)
{
  return (float)x;
}
__device__ float __uint2float_rz(unsigned int x)
{
  return (float)x;
}

__device__ float __uint_as_float(unsigned int x)
{
  hold32.ui = x;
  return hold32.f;
}

__device__ double __ull2double_rd(unsigned long long int x)
{
  return (double)x;
}
__device__ double __ull2double_rn(unsigned long long int x)
{
  return (double)x;
}
__device__ double __ull2double_ru(unsigned long long int x)
{
  return (double)x;
}
__device__ double __ull2double_rz(unsigned long long int x)
{
  return (double)x;
}

__device__ float __ull2float_rd(unsigned long long int x)
{
  return (float)x;
}
__device__ float __ull2float_rn(unsigned long long int x)
{
  return (float)x;
}
__device__ float __ull2float_ru(unsigned long long int x)
{
  return (float)x;
}
__device__ float __ull2float_rz(unsigned long long int x)
{
  return (float)x;
}

/*
Integer Intrinsics
*/

// integer intrinsic function __poc __clz __ffs __brev
__device__ unsigned int __popc( unsigned int input)
{
    return hc::__popcount_u32_b32(input);
}

__device__ unsigned int __popcll( unsigned long long int input)
{
    return hc::__popcount_u32_b64(input);
}

__device__ unsigned int __clz(unsigned int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 32 : hc::__firstbit_u32_u32( input);
#else
    return hc::__firstbit_u32_u32( input);
#endif
}

__device__ unsigned int __clzll(unsigned long long int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 64 : hc::__firstbit_u32_u64( input);
#else
    return hc::__firstbit_u32_u64( input);
#endif
}

__device__ unsigned int __clz( int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 32 : hc::__firstbit_u32_s32( input);
#else
    return hc::__firstbit_u32_s32( input);
#endif
}

__device__ unsigned int __clzll( long long int input)
{
#ifdef NVCC_COMPAT
    return input == 0 ? 64 : hc::__firstbit_u32_s64( input);
#else
    return hc::__firstbit_u32_s64( input);
#endif
}

__device__ unsigned int __ffs(unsigned int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_u32( input)+1;
#else
    return hc::__lastbit_u32_u32( input);
#endif
}

__device__ unsigned int __ffsll(unsigned long long int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_u64( input)+1;
#else
    return hc::__lastbit_u32_u64( input);
#endif
}

__device__ unsigned int __ffs( int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_s32( input)+1;
#else
    return hc::__lastbit_u32_s32( input);
#endif
}

__device__ unsigned int __ffsll( long long int input)
{
#ifdef NVCC_COMPAT
    return hc::__lastbit_u32_s64( input)+1;
#else
    return hc::__lastbit_u32_s64( input);
#endif
}

__device__ unsigned int __brev( unsigned int input)
{
    return hc::__bitrev_b32( input);
}

__device__ unsigned long long int __brevll( unsigned long long int input)
{
    return hc::__bitrev_b64( input);
}

struct ucharHolder {
  union {
    unsigned char c[4];
    unsigned int ui;
  };
}__attribute__((aligned(4)));

struct uchar2Holder {
  union {
    unsigned int ui[2];
    unsigned char c[8];
  };
}__attribute__((aligned(8)));

struct intHolder {
  union {
    signed int si[2];
    signed int long sl;
  };
}__attribute__((aligned(8)));

struct uintHolder {
  union {
    signed int ui[2];
    signed int long ul;
  };
}__attribute__((aligned(8)));


__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)
{
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

__device__ long long __mul64hi(long long int x, long long int y)
{
  struct intHolder iHold1;
  struct intHolder iHold2;
  iHold1.sl = x;
  iHold2.sl = y;
  iHold1.sl = iHold1.si[1] * iHold2.si[1];
  return iHold1.sl;
}

__device__ unsigned long long __umul64hi(unsigned long long int x, unsigned long long int y)
{
  struct uintHolder uHold1;
  struct uintHolder uHold2;
  uHold1.ul = x;
  uHold2.ul = y;
  uHold1.ul = uHold1.ui[1] * uHold2.ui[1];
  return uHold1.ul;
}

/*
HIP specific device functions
*/

__device__ unsigned __hip_ds_bpermute(int index, unsigned src) {
    return hc::__amdgcn_ds_bpermute(index, src);
}

__device__ float __hip_ds_bpermutef(int index, float src) {
    return hc::__amdgcn_ds_bpermute(index, src);
}

__device__ unsigned __hip_ds_permute(int index, unsigned src) {
    return hc::__amdgcn_ds_permute(index, src);
}

__device__ float __hip_ds_permutef(int index, float src) {
    return hc::__amdgcn_ds_permute(index, src);
}

__device__ unsigned __hip_ds_swizzle(unsigned int src, int pattern) {
    return hc::__amdgcn_ds_swizzle(src, pattern);
}

__device__ float __hip_ds_swizzlef(float src, int pattern) {
    return hc::__amdgcn_ds_swizzle(src, pattern);
}

__device__ int __hip_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl) {
    return hc::__amdgcn_move_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}

#define MASK1 0x00ff00ff
#define MASK2 0xff00ff00

__device__ char4 __hip_hc_add8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.a & MASK1;
    unsigned one2 = in2.a & MASK1;
    out.a = (one1 + one2) & MASK1;
    one1 = in1.a & MASK2;
    one2 = in2.a & MASK2;
    out.a = out.a | ((one1 + one2) & MASK2);
    return out;
}

__device__ char4 __hip_hc_sub8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.a & MASK1;
    unsigned one2 = in2.a & MASK1;
    out.a = (one1 - one2) & MASK1;
    one1 = in1.a & MASK2;
    one2 = in2.a & MASK2;
    out.a = out.a | ((one1 - one2) & MASK2);
    return out;
}

__device__ char4 __hip_hc_mul8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.a & MASK1;
    unsigned one2 = in2.a & MASK1;
    out.a = (one1 * one2) & MASK1;
    one1 = in1.a & MASK2;
    one2 = in2.a & MASK2;
    out.a = out.a | ((one1 * one2) & MASK2);
    return out;
}
