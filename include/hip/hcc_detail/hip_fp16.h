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

#ifndef HIP_FP16_H
#define HIP_FP16_H

#include "hip/hip_runtime.h"

using __half = __fp16;

struct __half2 {
    __half p;
    __half q;
};

struct half {
    __host__ __device__
    operator __half() const { return x; }

    __half x;
};

typedef __half2 half2;

/*
Arithmetic functions
*/

__device__
static
inline
__half __saturate(__half a)
{
    return (a < 0) ? 0 : ((a > 1) ? 1 : a);
}

__device__
static
inline
__half __hadd(__half a, __half b)
{
    return a + b;
}

__device__
static
inline
__half __hadd_sat(__half a, __half b)
{
    return __saturate(__hadd(a, b));
}

__device__
static
inline
__half __hfma(__half a, __half b, __half c)
{
    return fmaf(a, b, c);
}

__device__
static
inline
__half __hfma_sat(__half a, __half b, __half c)
{
    return __saturate(__hfma(a, b, c));
}

__device__
static
inline
__half __hmul(__half a, __half b)
{
    return a * b;
}

__device__
static
inline
__half __hmul_sat(__half a, __half b)
{
    return __saturate(__hmul(a, b));
}

__device__
static
inline
__half __hneg(__half a)
{
    return -a;
}

__device__
static
inline
__half __hsub(__half a, __half b)
{
    return a - b;
}

__device__
static
inline
__half __hsub_sat(__half a, __half b)
{
    return __saturate(__hsub(a, b));
}

__device__
static
inline
__half hdiv(__half a, __half b) { return a / b; }


/*
Half2 Arithmetic Instructions
*/

__device__
static
inline
__half2 __hadd2(__half2 a, __half2 b)
{
    return {__hadd(a.p, b.p), __hadd(a.q, b.q)};
}

__device__
static
inline
__half2 __hadd2_sat(__half2 a, __half2 b)
{
    return {__hadd_sat(a.p, b.p), __hadd_sat(a.q, b.q)};
}

__device__
static
inline
__half2 __hfma2(__half2 a, __half2 b, __half2 c)
{
    return {__hfma(a.p, b.p, c.p), __hfma(a.q, b.q, c.q)};
}

__device__
static
inline
__half2 __hfma2_sat(__half2 a, __half2 b, __half2 c)
{
    return {__hfma_sat(a.p, b.p, c.p), __hfma_sat(a.q, b.q, c.q)};
}

__device__
static
inline
__half2 __hmul2(__half2 a, __half2 b)
{
    return {__hmul(a.p, b.p), __hmul(a.q, b.q)};
}

__device__
static
inline
__half2 __hmul2_sat(__half2 a, __half2 b)
{
    return {__hmul_sat(a.p, b.p), __hmul_sat(a.q, b.q)};
}

__device__
static
inline
__half2 __hneg2(__half2 a)
{
    return {__hneg(a.p), __hneg(a.q)};
}

__device__
static
inline
__half2 __hsub2(__half2 a, __half2 b)
{
    return {__hsub(a.p, b.p), __hsub(a.q, b.q)};
}

__device__
static
inline
__half2 __hsub2_sat(__half2 a, __half2 b)
{
    return {__hsub_sat(a.p, b.p), __hsub_sat(a.q, b.q)};
}

__device__
static
inline
__half2 h2div(__half2 a, __half2 b)
{
    return {hdiv(a.p, b.p), hdiv(a.q, b.q)};
}

/*
Half Cmps
*/

__device__
static
inline
bool __heq(__half a, __half b)
{
    return a == b;
}

__device__
static
inline
bool __hge(__half a, __half b){
    return !(a < b);
}

__device__
static
inline
bool __hgt(__half a, __half b){
    return b < a;
}

__device__
static
inline
bool __hisinf(__half a)
{
    constexpr __half __half_pos_inf = 0x7C00;
    constexpr __half __half_neg_inf = 0xFC00;
    return ((a == __half_neg_inf) ? -1 : (a == __half_pos_inf) ? 1 : 0);
}

__device__
static
inline
bool __hisnan(__half a)
{
    return a != a;
}

__device__
static
inline
bool __hle(__half a, __half b)
{
    return !(b < a);
}

__device__
static
inline
bool __hlt(__half a, __half b)
{
    return a < b;
}

__device__
static
inline
bool __hne(__half a, __half b)
{
    return !(a == b);
}

/*
Half2 Cmps
*/

__device__
static
inline
bool __hbeq2(__half2 a, __half2 b)
{
    return __heq(a.p, b.p) && __heq(a.q, b.q);
}

__device__
static
inline
bool __hbge2(__half2 a, __half2 b)
{
    return __hge(a.p, b.p) && __hge(a.q, b.q);
}

__device__
static
inline
bool __hbgt2(__half2 a, __half2 b)
{
    return __hgt(a.p, b.p) && __hgt(a.q, b.q);
}

__device__
static
inline
bool __hble2(__half2 a, __half2 b)
{
    return __hle(a.p, b.p) && __hle(a.q, b.q);
}

__device__
static
inline
bool __hblt2(__half2 a, __half2 b)
{
    return __hlt(a.p, b.p) && __hlt(a.q, b.q);
}

__device__
static
inline
bool __hbne2(__half2 a, __half2 b)
{
    return __hne(a.p, b.p) && __hne(a.q, b.q);
}

__device__
static
inline
__half2 __heq2(__half2 a, __half2 b)
{
    return {static_cast<__half>(__heq(a.p, b.p)),
            static_cast<__half>(__heq(a.q, b.q))};
}

__device__
static
inline
__half2 __hge2(__half2 a, __half2 b)
{
    return {static_cast<__half>(__hge(a.p, b.p)),
            static_cast<__half>(__hge(a.q, b.q))};
}

__device__
static
inline
__half2 __hgt2(__half2 a, __half2 b)
{
    return {static_cast<__half>(__hgt(a.p, b.p)),
            static_cast<__half>(__hgt(a.q, b.q))};
}

__device__
static
inline
__half2 __hisnan2(__half2 a)
{
    return {static_cast<__half>(__hisnan(a.p)),
            static_cast<__half>(__hisnan(a.q))};
}

__device__
static
inline
__half2 __hle2(__half2 a, __half2 b)
{
    return {static_cast<__half>(__hle(a.p, b.p)),
            static_cast<__half>(__hle(a.q, b.q))};
}

__device__
static
inline
__half2 __hlt2(__half2 a, __half2 b)
{
    return {static_cast<__half>(__hlt(a.p, b.p)),
            static_cast<__half>(__hlt(a.q, b.q))};
}

__device__
static
inline
__half2 __hne2(__half2 a, __half2 b)
{
    return {static_cast<__half>(__hne(a.p, b.p)),
            static_cast<__half>(__hne(a.q, b.q))};
}

/*
Half Cnvs and Data Mvmnt
*/

__device__
static
inline
__half2 __float22half2_rn(float2 a)
{
    return {static_cast<__half>(a.x), static_cast<__half>(a.y)};
}

__device__
static
inline
__half __float2half(float a)
{
    return a;
}

__device__
static
inline
__half2 __float2half2_rn(float a)
{
    return {static_cast<__half>(a), static_cast<__half>(a)};
}

__device__
static
inline
__half2 __floats2half2_rn(float a, float b)
{
    return {static_cast<__half>(a), static_cast<__half>(b)};
}

__device__
static
inline
float2 __half22float2(__half2 a)
{
    return {a.p, a.q};
}

__device__
static
inline
float __half2float(__half a)
{
    return a;
}

__device__
static
inline
__half2 __half2half2(__half a)
{
    return {a, a};
}

__device__
static
inline
__half2 __halves2half2(__half a, __half b)
{
    return {a, b};
}

__device__
static
inline
float __high2float(__half2 a)
{
    return a.p;
}

__device__
static
inline
__half __high2half(__half2 a)
{
    return a.p;
}

__device__
static
inline
__half2 __high2half2(__half2 a)
{
    return {a.p, a.p};
}

__device__
static
inline
__half2 __highs2half2(__half2 a, __half2 b)
{
    return {a.p, b.p};
}

__device__
static
inline
float __low2float(__half2 a)
{
    return a.q;
}

__device__
static
inline
__half __low2half(__half2 a)
{
    return a.q;
}

__device__
static
inline
__half2 __low2half2(__half2 a)
{
    return {a.q, a.q};
}

__device__
static
inline
__half2 __lows2half2(__half2 a, __half2 b)
{
    return {a.q, b.q};
}

__device__
static
inline
__half2 __lowhigh2highlow(__half2 a)
{
    return {a.q, a.p};
}

__device__
static
inline
__half2 __low2half2(__half2 a, __half2 b)
{
    return {a.q, b.q};
}

#endif
