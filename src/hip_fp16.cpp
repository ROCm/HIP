/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include"hip/hip_fp16.h"

static const unsigned sign_val = 0x8000;
static const __half __half_value_one_float = {0x3C00};
static const __half __half_value_zero_float = {0x0};
static const unsigned __half_pos_inf = 0x7C00;
static const unsigned __half_neg_inf = 0xFC00;

typedef struct{
  union{
    float f;
    unsigned u;
  };
} struct_float;

#if __clang_major__ == 3

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


__device__ __half __soft_hadd(const __half a, const __half b){
  return cvt_float_to_half(cvt_half_to_float(a)+cvt_half_to_float(b));
}

__device__ __half __soft_hadd_sat(const __half a, const __half b){
  float f = cvt_half_to_float(a) + cvt_half_to_float(b);
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}

__device__ __half __soft_hfma(const __half a, const __half b, const __half c){
  return cvt_float_to_half(fmaf(cvt_half_to_float(a), cvt_half_to_float(b), cvt_half_to_float(c)));
}

__device__ __half __soft_hfma_sat(const __half a, const __half b, const __half c){
  float f = fmaf(cvt_half_to_float(a), cvt_half_to_float(b), cvt_half_to_float(c));
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}

__device__ __half __soft_hmul(const __half a, const __half b){
  return cvt_float_to_half(cvt_half_to_float(a)*cvt_half_to_float(b));
}

__device__ __half __soft_hmul_sat(const __half a, const __half b){
  float f = cvt_half_to_float(a) * cvt_half_to_float(b);
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}

__device__ __half __soft_hneq(const __half a){
  __half ret = {a.x};
  ret.x ^= 1 << 15;
  return ret;
}

__device__ __half __soft_hsub(const __half a, const __half b){
  return cvt_float_to_half(cvt_half_to_float(a)-cvt_half_to_float(b));
}

__device__ __half __soft_hsub_sat(const __half a, const __half b){
  float f = cvt_half_to_float(a) - cvt_half_to_float(b);
  return (f < 0.0f ? __half_value_zero_float : (f > 1.0f ? __half_value_one_float: cvt_float_to_half(f)));
}


/*
Half2 Arithmetic Instructions
*/

__device__ __half2 __soft_hadd2(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p[1] = __soft_hadd(a.p[1], b.p[1]);
  ret.p[0] = __soft_hadd(a.p[0], b.p[0]);
  return ret;
}

__device__ __half2 __soft_hadd2_sat(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p[1] = __soft_hadd_sat(a.p[1], b.p[1]);
  ret.p[0] = __soft_hadd_sat(a.p[0], b.p[0]);
  return ret;
}

__device__ __half2 __soft_hfma2(const __half2 a, const __half2 b, const __half2 c){
  __half2 ret;
  ret.p[1] = __soft_hfma(a.p[1], b.p[1], c.p[1]);
  ret.p[0] = __soft_hfma(a.p[0], b.p[0], c.p[0]);
  return ret;
}

__device__ __half2 __soft_hfma2_sat(const __half2 a, const __half2 b, const __half2 c){
  __half2 ret;
  ret.p[1] = __soft_hfma_sat(a.p[1], b.p[1], c.p[1]);
  ret.p[0] = __soft_hfma_sat(a.p[0], b.p[0], c.p[0]);
  return ret;
}

__device__ __half2 __soft_hmul2(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p[1] = __soft_hmul(a.p[1], b.p[1]);
  ret.p[0] = __soft_hmul(a.p[0], b.p[0]);
  return ret;
}

__device__ __half2 __soft_hmul2_sat(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p[1] = __soft_hmul_sat(a.p[1], b.p[1]);
  ret.p[0] = __soft_hmul_sat(a.p[0], b.p[0]);
  return ret;
}

__device__ __half2 __soft_hneq2(const __half2 a){
  __half2 ret;
  ret.p[1] = __soft_hneq(a.p[1]);
  ret.p[0] = __soft_hneq(a.p[0]);
  return ret;
}

__device__ __half2 __soft_hsub2(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p[1] = __soft_hsub(a.p[1], b.p[1]);
  ret.p[0] = __soft_hsub(a.p[0], b.p[0]);
  return ret;
}

__device__ __half2 __soft_hsub2_sat(const __half2 a, const __half2 b){
  __half2 ret;
  ret.p[1] = __soft_hsub_sat(a.p[1], b.p[1]);
  ret.p[0] = __soft_hsub_sat(a.p[0], b.p[0]);
  return ret;
}

/*
Half Cmps
*/

__device__  bool __soft_heq(const __half a, const __half b){
  return (a.x == b.x ? true:false);
}

__device__ bool __soft_hge(const __half a, const __half b){
  return (cvt_half_to_float(a) >= cvt_half_to_float(b));
}

__device__ bool __soft_hgt(const __half a, const __half b){
  return (cvt_half_to_float(a) > cvt_half_to_float(b));
}

__device__ bool __soft_hisinf(const __half a){
  return ((a.x == __half_neg_inf) ? -1 : (a.x == __half_pos_inf) ? 1 : 0);
}

__device__ bool __soft_hisnan(const __half a){
  if(((a.x & __half_pos_inf) == a.x) || ((a.x & __half_neg_inf) == a.x)){
    return true;
  }else{
    return false;
  }
}

__device__ bool __soft_hle(const __half a, const __half b){
  return (cvt_half_to_float(a) <= cvt_half_to_float(b));
}

__device__ bool __soft_hlt(const __half a, const __half b){
  return (cvt_half_to_float(a) < cvt_half_to_float(b));
}

__device__ bool __soft_hne(const __half a, const __half b){
  return a.x == b.x ? false : true;
}

/*
Half2 Cmps
*/

__device__ bool __soft_hbeq2(const __half2 a, const __half2 b){
  return __soft_heq(a.p[1], b.p[1]) && __soft_heq(a.p[0], b.p[0]);
}

__device__ bool __soft_hbge2(const __half2 a, const __half2 b){
  return __soft_hge(a.p[1], b.p[1]) && __soft_hge(a.p[0], b.p[0]);
}

__device__ bool __soft_hbgt2(const __half2 a, const __half2 b){
  return __soft_hgt(a.p[1], b.p[1]) && __soft_hgt(a.p[0], b.p[0]);
}

__device__ bool __soft_hble2(const __half2 a, const __half2 b){
  return __soft_hle(a.p[1], b.p[1]) && __soft_hle(a.p[0], b.p[0]);
}

__device__ bool __soft_hblt2(const __half2 a, const __half2 b){
  return __soft_hlt(a.p[1], b.p[1]) && __soft_hlt(a.p[0], b.p[0]);
}

__device__ bool __soft_hbne2(const __half2 a, const __half2 b){
  return __soft_hne(a.p[1], b.p[1]) && __soft_hne(a.p[0], b.p[0]);
}



__device__ __half2 __soft_heq2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p[1] = (__soft_heq(a.p[1], b.p[1])) ? __half_value_one_float : __half_value_zero_float;
  ret.p[0] = (__soft_heq(a.p[0], b.p[0])) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __soft_hge2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p[1] = (__soft_hge(a.p[1], b.p[1])) ? __half_value_one_float : __half_value_zero_float;
  ret.p[0] = (__soft_hge(a.p[0], b.p[0])) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __soft_hgt2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p[1] = (__soft_hgt(a.p[1], b.p[1])) ? __half_value_one_float : __half_value_zero_float;
  ret.p[0] = (__soft_hgt(a.p[0], b.p[0])) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __soft_hisnan2(const __half2 a){
  __half2 ret = {0};
  ret.p[1] = __soft_hisnan(a.p[1]) ? __half_value_one_float : __half_value_zero_float;
  ret.p[0] = __soft_hisnan(a.p[0]) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __soft_hle2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p[1] = (__soft_hle(a.p[1], b.p[1])) ? __half_value_one_float : __half_value_zero_float;
  ret.p[0] = (__soft_hle(a.p[0], b.p[0])) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __soft_hlt2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p[1] = (__soft_hlt(a.p[1], b.p[1])) ? __half_value_one_float : __half_value_zero_float;
  ret.p[0] = (__soft_hlt(a.p[0], b.p[0])) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

__device__ __half2 __soft_hne2(const __half2 a, const __half2 b){
  __half2 ret = {0};
  ret.p[1] = (__soft_hne(a.p[1], b.p[1])) ? __half_value_one_float : __half_value_zero_float;
  ret.p[0] = (__soft_hne(a.p[0], b.p[0])) ? __half_value_one_float : __half_value_zero_float;
  return ret;
}

/*
Half Cnvs and Data Mvmnt
*/

__device__ __half2 __soft_float22half2_rn(const float2 a){
  __half2 ret = {0};
  ret.p[1] = cvt_float_to_half(a.x);
  ret.p[0] = cvt_float_to_half(a.y);
  return ret;
}

__device__ __half __soft_float2half(const float a){
  return cvt_float_to_half(a);
}

__device__ __half2 __soft_float2half2_rn(const float a){
  __half ret = cvt_float_to_half(a);
  return {ret, ret};
}

__device__ __half2 __soft_floats2half2_rn(const float a, const float b){
  return {cvt_float_to_half(a), cvt_float_to_half(b)};
}

__device__ float2 __soft_half22float2(const __half2 a){
  return {cvt_half_to_float(a.p[1]), cvt_half_to_float(a.p[0])};
}

__device__ float __soft_half2float(const __half a){
  return cvt_half_to_float(a);
}

__device__ __half2 __soft_half2half2(const __half a){
  return {a,a};
}

__device__ __half2 __soft_halves2half2(const __half a, const __half b){
  return {a,b};
}

__device__ float __soft_high2float(const __half2 a){
  return cvt_half_to_float(a.p[1]);
}

__device__ __half __soft_high2half(const __half2 a){
  return a.p[1];
}

__device__ __half2 __soft_high2half2(const __half2 a){
  return {a.p[1], a.p[1]};
}

__device__ __half2 __soft_highs2half2(const __half2 a, const __half2 b){
  return {a.p[1], b.p[1]};
}

__device__ float __soft_low2float(const __half2 a){
  return cvt_half_to_float(a.p[0]);
}

__device__ __half __soft_low2half(const __half2 a){
  return a.p[0];
}

__device__ __half2 __soft_low2half2(const __half2 a){
  return {a.p[0], a.p[0]};
}

__device__ __half2 __soft_lows2half2(const __half2 a, const __half2 b){
  return {a.p[0], b.p[0]};
}

__device__ __half2 __soft_lowhigh2highlow(const __half2 a){
  return {a.p[0], a.p[1]};
}

__device__ __half2 __soft_low2half2(const __half2 a, const __half2 b){
  return {a.p[0], b.p[0]};
}

#endif
