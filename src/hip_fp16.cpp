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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include"hip/hip_fp16.h"

static const unsigned sign_val = 0x8000;
static const __half __half_value_one_float = {0x3C00};
static const __half __half_value_zero_float = {0x0};
static const unsigned __half_pos_inf = 0x7C00;
static const unsigned __half_neg_inf = 0xFC00;

static __device__ float cvt_half_to_float(__half a){
  struct_float ret = {0};
  if(a.x == 0){
    return 0.0f;
  }
  if(a.x == 0x8000){
    return -0.0f;
  }
  ret.u = ((a.x&0x8000)<<16) | (((a.x&0x7c00)+0x1C000)<<13) | ((a.x&0x03FF)<<13);
  return ret.f;
}

static __device__ __half cvt_float_to_half(float b){
  struct_float f = {0};
  __half ret = {0};
  f.f = b;
  if(f.f == 0.0f){
    ret.x = 0;
    return ret;
  }
  if(f.f == -0.0f){
    ret.x = 0x8000;
    return ret;
  }
  ret.x = ((f.u>>16)&0x8000)|((((f.u&0x7f800000)-0x38000000)>>13)&0x7c00)|((f.u>>13)&0x03ff);
  return ret;
}


__device__ __half __hadd(const __half a, const __half b){
  return cvt_float_to_half(cvt_half_to_float(a)+cvt_half_to_float(b));
}

__device__ __half __hadd_sat(const __half a, const __half b){
  float f = cvt_half_to_float(a) + cvt_half_to_float(b);
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}

__device__ __half __hfma(const __half a, const __half b, const __half c){
  return cvt_float_to_half(fmaf(cvt_half_to_float(a), cvt_half_to_float(b), cvt_half_to_float(c)));
}

__device__ __half __hfma_sat(const __half a, const __half b, const __half c){
  float f = fmaf(cvt_half_to_float(a), cvt_half_to_float(b), cvt_half_to_float(c));
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}

__device__ __half __hmul(const __half a, const __half b){
  return cvt_float_to_half(cvt_half_to_float(a)*cvt_half_to_float(b));
}

__device__ __half __hmul_sat(const __half a, const __half b){
  float f = cvt_half_to_float(a) * cvt_half_to_float(b);
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}

__device__ __half __hneq(const __half a){
  __half ret = {a.x};
  ret.x ^= 1 << 15;
  return ret;
}

__device__ __half __hsub(const __half a, const __half b){
  return cvt_float_to_half(cvt_half_to_float(a)-cvt_half_to_float(b));
}

__device__ __half __hsub_sat(const __half a, const __half b){
  float f = cvt_half_to_float(a) - cvt_half_to_float(b);
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}


/*
Half2 Arithmetic Instructions
*/

__device__ __half2 __hadd2(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p = __hadd(a.p, b.p);
  ret.q = __hadd(a.q, b.q);
  return ret;
}

__device__ __half2 __hadd2_sat(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p = __hadd_sat(a.p, b.p);
  ret.q = __hadd_sat(a.q, b.q);
  return ret;
}

__device__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c){
  __half2 ret;
  ret.p = __hfma(a.p, b.p, c.p);
  ret.q = __hfma(a.q, b.q, c.q);
  return ret;
}

__device__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c){
  __half2 ret;
  ret.p = __hfma_sat(a.p, b.p, c.p);
  ret.q = __hfma_sat(a.q, b.q, c.q);
  return ret;
}

__device__ __half2 __hmul2(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p = __hmul(a.p, b.p);
  ret.q = __hmul(a.q, b.q);
  return ret;
}

__device__ __half2 __hmul2_sat(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p = __hmul_sat(a.p, b.p);
  ret.q = __hmul_sat(a.q, b.q);
  return ret;
}

__device__ __half2 __hneq2(const __half2 a){
  __half2 ret;
  ret.p = __hneq(a.p);
  ret.q = __hneq(a.q);
  return ret;
}

__device__ __half2 __hsub2(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p = __hsub(a.p, b.p);
  ret.q = __hsub(a.q, b.q);
  return ret;
}

__device__ __half2 __hsub2_sat(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p = __hsub_sat(a.p, b.p);
  ret.q = __hsub_sat(a.q, b.q);
  return ret;
}

/*
Half Cmps
*/

__device__  bool __heq(const __half a, const __half b){
  return (a.x == b.x ? true:false);
}

__device__ bool __hge(const __half a, const __half b){
  return (cvt_half_to_float(a) >= cvt_half_to_float(b));
}

__device__ bool __hgt(const __half a, const __half b){
  return (cvt_half_to_float(a) > cvt_half_to_float(b));
}

__device__ bool __hisinf(const __half a){
  return ((a.x == __half_neg_inf) ? -1 : (a.x == __half_pos_inf) ? 1 : 0);
}

__device__ bool __hisnan(const __half a){
  if(((a.x & __half_pos_inf) == a.x) || ((a.x & __half_neg_inf) == a.x)){
    return true;
  }else{
    return false;
  }
}

__device__ bool __hle(const __half a, const __half b){
  return (cvt_half_to_float(a) <= cvt_half_to_float(b));
}

__device__ bool __hlt(const __half a, const __half b){
  return (cvt_half_to_float(a) < cvt_half_to_float(b));
}

__device__ bool __hne(const __half a, const __half b){
  return a.x == b.x ? false : true;
}

/*
Half2 Cmps
*/

__device__ bool __hbeq2(const __half2 a, const __half2 b){
  return __heq(a.p, b.p) && __heq(a.q, b.q);
}

__device__ bool __hbge2(const __half2 a, const __half2 b){
  return __hge(a.p, b.p) && __hge(a.q, b.q);
}

__device__ bool __hbgt2(const __half2 a, const __half2 b){
  return __hgt(a.p, b.p) && __hgt(a.q, b.q);
}

__device__ bool __hble2(const __half2 a, const __half2 b){
  return __hle(a.p, b.p) && __hle(a.q, b.q);
}

__device__ bool __hblt2(const __half2 a, const __half2 b){
  return __hlt(a.p, b.p) && __hlt(a.q, b.q);
}

__device__ bool __hbne2(const __half2 a, const __half2 b){
  return __hne(a.p, b.p) && __hne(a.q, b.q);
}



__device__ __half2 __heq2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p = (__heq(a.p, b.p)) ? __half_value_one_float : __half_value_zero_float;
  ret.q = (__heq(a.q, b.q)) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __hge2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p = (__hge(a.p, b.p)) ? __half_value_one_float : __half_value_zero_float;
  ret.q = (__hge(a.q, b.q)) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __hgt2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p = (__hgt(a.p, b.p)) ? __half_value_one_float : __half_value_zero_float;
  ret.q = (__hgt(a.q, b.q)) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __hisnan2(const __half2 a){
  __half2 ret = {0};
  ret.p = __hisnan(a.p) ? __half_value_one_float : __half_value_zero_float;
  ret.q = __hisnan(a.q) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __hle2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p = (__hle(a.p, b.p)) ? __half_value_one_float : __half_value_zero_float;
  ret.q = (__hle(a.q, b.q)) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __hlt2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p = (__hlt(a.p, b.p)) ? __half_value_one_float : __half_value_zero_float;
  ret.q = (__hlt(a.q, b.q)) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __hne2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p = (__hne(a.p, b.p)) ? __half_value_one_float : __half_value_zero_float;
  ret.q = (__hne(a.q, b.q)) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

/*
Half Cnvs and Data Mvmnt
*/

__device__ __half2 __float22half2_rn(const float2 a){
  __half2 ret = {0};
  ret.p = cvt_float_to_half(a.x);
  ret.q = cvt_float_to_half(a.y);
  return ret;
}

__device__ __half __float2half(const float a){
  return cvt_float_to_half(a);
}

__device__ __half2 __float2half2_rn(const float a){
  __half ret = cvt_float_to_half(a);
  return {ret, ret};
}

__device__ __half2 __floats2half2_rn(const float a, const float b){
  return {cvt_float_to_half(a), cvt_float_to_half(b)};
}

__device__ float2 __half22float2(const __half2 a){
  return {cvt_half_to_float(a.p), cvt_half_to_float(a.q)};
}

__device__ float __half2float(const __half a){
  return cvt_half_to_float(a);
}

__device__ __half2 __half2half2(const __half a){
  return {a,a};
}

__device__ __half2 __halves2half2(const __half a, const __half b){
  return {a,b};
}

__device__ float __high2float(const __half2 a){
  return cvt_half_to_float(a.p);
}

__device__ __half __high2half(const __half2 a){
  return a.p;
}

__device__ __half2 __high2half2(const __half2 a){
  return {a.p, a.p};
}

__device__ __half2 __highs2half2(const __half2 a, const __half2 b){
  return {a.p, b.p};
}

__device__ float __low2float(const __half2 a){
  return cvt_half_to_float(a.q);
}

__device__ __half __low2half(const __half2 a){
  return a.q;
}

__device__ __half2 __low2half2(const __half2 a){
  return {a.q, a.q};
}

__device__ __half2 __lows2half2(const __half2 a, const __half2 b){
  return {a.q, b.q};
}

__device__ __half2 __lowhigh2highlow(const __half2 a){
  return {a.q, a.p};
}

__device__ __half2 __low2half2(const __half2 a, const __half2 b){
  return {a.q, b.q};
}

