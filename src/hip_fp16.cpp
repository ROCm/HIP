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

#include"hip/hcc_detail/hip_fp16.h"

struct hipHalfHolder{
  union {
    __half h;
    unsigned short s;
  };
};

#define HINF 65504

__device__ static struct hipHalfHolder __hInfValue = {HINF};

__device__ __half __hadd(__half a, __half b) {
  return a + b;
}

__device__ __half __hadd_sat(__half a, __half b) {
  return a + b;
}

__device__ __half __hfma(__half a, __half b, __half c) {
  return a * b + c;
}

__device__ __half __hfma_sat(__half a, __half b, __half c) {
  return a * b + c;
}

__device__ __half __hmul(__half a, __half b) {
  return a * b;
}

__device__ __half __hmul_sat(__half a, __half b) {
  return a * b;
}

__device__ __half __hneg(__half a) {
  return -a;
}

__device__ __half __hsub(__half a, __half b) {
  return a - b;
}

__device__ __half __hsub_sat(__half a, __half b) {
  return a - b;
}

__device__ __half hdiv(__half a, __half b) {
  return a / b;
}

/*
Half comparision Functions
*/

__device__  bool __heq(__half a, __half b) {
  return a == b ? true : false;
}

__device__  bool __hge(__half a, __half b) {
  return a >= b ? true : false;
}

__device__  bool __hgt(__half a, __half b) {
  return a > b ? true : false;
}

__device__  bool __hisinf(__half a) {
  return a == HINF ? true : false;
}

__device__  bool __hisnan(__half a) {
  return a > HINF ? true : false;
}

__device__  bool __hle(__half a, __half b) {
  return a <= b ? true : false;
}

__device__  bool __hlt(__half a, __half b) {
  return a < b ? true : false;
}

__device__  bool __hne(__half a, __half b) {
  return a != b ? true : false;
}

/*
Half2 Comparision Functions
*/

__device__  bool __hbeq2(__half2 a, __half2 b) {
  return (a.x == b.x ? true : false) && (a.y == b.y ? true : false);
}

__device__  bool __hbge2(__half2 a, __half2 b) {
  return (a.x >= b.x ? true : false) && (a.y >= b.y ? true : false);
}

__device__  bool __hbgt2(__half2 a, __half2 b) {
  return (a.x > b.x ? true : false) && (a.y > b.y ? true : false);
}

__device__  bool __hble2(__half2 a, __half2 b) {
  return (a.x <= b.x ? true : false) && (a.y <= b.y ? true : false);
}

__device__  bool __hblt2(__half2 a, __half2 b) {
  return (a.x < b.x ? true : false) && (a.y < b.y ? true : false);
}

__device__  bool __hbne2(__half2 a, __half2 b) {
  return (a.x != b.x ? true : false) && (a.y != b.y ? true : false);
}

__device__  __half2 __heq2(__half2 a, __half2 b) {
  __half2 c;
  c.x = (a.x == b.x) ? (__half)1 : (__half)0;
  c.y = (a.y == b.y) ? (__half)1 : (__half)0;
  return c;
}

__device__  __half2 __hge2(__half2 a, __half2 b) {
  __half2 c;
  c.x = (a.x >= b.x) ? (__half)1 : (__half)0;
  c.y = (a.y >= b.y) ? (__half)1 : (__half)0;
  return c;
}

__device__  __half2 __hgt2(__half2 a, __half2 b) {
  __half2 c;
  c.x = (a.x > b.x) ? (__half)1 : (__half)0;
  c.y = (a.y > b.y) ? (__half)1 : (__half)0;
  return c;
}

__device__  __half2 __hisnan2(__half2 a) {
  __half2 c;
  c.x = (a.x > HINF) ? (__half)1 : (__half)0;
  c.y = (a.y > HINF) ? (__half)1 : (__half)0;
  return c;
}

__device__  __half2 __hle2(__half2 a, __half2 b) {
  __half2 c;
  c.x = (a.x <= b.x) ? (__half)1 : (__half)0;
  c.y = (a.y <= b.y) ? (__half)1 : (__half)0;
  return c;
}

__device__  __half2 __hlt2(__half2 a, __half2 b) {
  __half2 c;
  c.x = (a.x < b.x) ? (__half)1 : (__half)0;
  c.y = (a.y < b.y) ? (__half)1 : (__half)0;
  return c;
}

__device__  __half2 __hne2(__half2 a, __half2 b) {
  __half2 c;
  c.x = (a.x != b.x) ? (__half)1 : (__half)0;
  c.y = (a.y != b.y) ? (__half)1 : (__half)0;
  return c;
}

/*
Conversion instructions
*/
__device__  __half2 __float22half2_rn(const float2 a) {
  __half2 b;
  b.x = (__half)a.x;
  b.y = (__half)a.y;
  return b;
}

__device__  __half __float2half(const float a) {
  return (__half)a;
}

__device__  __half2 __float2half2_rn(const float a) {
  __half2 b;
  b.x = (__half)a;
  b.y = (__half)a;
  return b;
}

__device__  __half __float2half_rd(const float a) {
  return (__half)a;
}

__device__  __half __float2half_rn(const float a) {
  return (__half)a;
}

__device__  __half __float2half_ru(const float a) {
  return (__half)a;
}

__device__  __half __float2half_rz(const float a) {
  return (__half)a;
}

__device__  __half2 __floats2half2_rn(const float a, const float b) {
  __half2 c;
  c.x = (__half)a;
  c.y = (__half)b;
  return c;
}

__device__  float2 __half22float2(const __half2 a) {
  float2 b;
  b.x = (float)a.x;
  b.y = (float)a.y;
  return b;
}

__device__  float __half2float(const __half a) {
  return (float)a;
}

__device__  __half2 half2half2(const __half a) {
  __half2 b;
  b.x = a;
  b.y = a;
  return b;
}

__device__  int __half2int_rd(__half h) {
  return (int)h;
}

__device__  int __half2int_rn(__half h) {
  return (int)h;
}

__device__  int __half2int_ru(__half h) {
  return (int)h;
}

__device__  int __half2int_rz(__half h) {
  return (int)h;
}

__device__  long long int __half2ll_rd(__half h) {
  return (long long int)h;
}

__device__  long long int __half2ll_rn(__half h) {
  return (long long int)h;
}

__device__  long long int __half2ll_ru(__half h) {
  return (long long int)h;
}

__device__  long long int __half2ll_rz(__half h) {
  return (long long int)h;
}

__device__  short __half2short_rd(__half h) {
  return (short)h;
}

__device__  short __half2short_rn(__half h) {
  return (short)h;
}

__device__  short __half2short_ru(__half h) {
  return (short)h;
}

__device__  short __half2short_rz(__half h) {
  return (short)h;
}

__device__  unsigned int __half2uint_rd(__half h) {
  return (unsigned int)h;
}

__device__  unsigned int __half2uint_rn(__half h) {
  return (unsigned int)h;
}

__device__  unsigned int __half2uint_ru(__half h) {
  return (unsigned int)h;
}

__device__  unsigned int __half2uint_rz(__half h) {
  return (unsigned int)h;
}

__device__  unsigned long long int __half2ull_rd(__half h) {
  return (unsigned long long)h;
}

__device__  unsigned long long int __half2ull_rn(__half h) {
  return (unsigned long long)h;
}

__device__  unsigned long long int __half2ull_ru(__half h) {
  return (unsigned long long)h;
}

__device__  unsigned long long int __half2ull_rz(__half h) {
  return (unsigned long long)h;
}

__device__  unsigned short int __half2ushort_rd(__half h) {
  return (unsigned short int)h;
}

__device__  unsigned short int __half2ushort_rn(__half h) {
  return (unsigned short int)h;
}

__device__  unsigned short int __half2ushort_ru(__half h) {
  return (unsigned short int)h;
}

__device__  unsigned short int __half2ushort_rz(__half h) {
  return (unsigned short int)h;
}

__device__  short int __half_as_short(const __half h) {
  hipHalfHolder hH;
  hH.h = h;
  return (short)hH.s;
}

__device__  unsigned short int __half_as_ushort(const __half h) {
  hipHalfHolder hH;
  hH.h = h;
  return hH.s;
}

__device__  __half2 __halves2half2(const __half a, const __half b) {
  __half2 c;
  c.x = a;
  c.y = b;
  return c;
}

__device__  float __high2float(const __half2 a) {
  return (float)a.y;
}

__device__  __half __high2half(const __half2 a) {
  return a.y;
}

__device__  __half2 __high2half2(const __half2 a) {
  __half2 b;
  b.x = a.y;
  b.y = a.y;
  return b;
}

__device__  __half2 __highs2half2(const __half2 a, const __half2 b) {
  __half2 c;
  c.x = a.y;
  c.y = b.y;
  return c;
}

__device__  __half __int2half_rd(int i) {
  return (__half)i;
}

__device__  __half __int2half_rn(int i) {
  return (__half)i;
}

__device__  __half __int2half_ru(int i) {
  return (__half)i;
}

__device__  __half __int2half_rz(int i) {
  return (__half)i;
}

__device__  __half __ll2half_rd(long long int i){
  return (__half)i;
}

__device__  __half __ll2half_rn(long long int i){
  return (__half)i;
}

__device__  __half __ll2half_ru(long long int i){
  return (__half)i;
}

__device__  __half __ll2half_rz(long long int i){
  return (__half)i;
}

__device__  float __low2float(const __half2 a) {
  return (float)a.x;
}

__device__  __half __low2half(const __half2 a) {
  return a.x;
}

__device__  __half2 __low2half2(const __half2 a, const __half2 b) {
  __half2 c;
  c.x = a.x;
  c.y = b.x;
  return c;
}

__device__  __half2 __low2half2(const __half2 a) {
  __half2 b;
  b.x = a.x;
  b.y = a.x;
  return b;
}

__device__  __half2 __lowhigh2highlow(const __half2 a) {
  __half2 b;
  b.x = a.y;
  b.y = a.x;
  return b;
}

__device__  __half2 __lows2half2(const __half2 a, const __half2 b) {
  __half2 c;
  c.y = a.x;
  c.y = b.x;
  return c;
}

__device__  __half __short2half_rd(short int i) {
  return (__half)i;
}

__device__  __half __short2half_rn(short int i) {
  return (__half)i;
}

__device__  __half __short2half_ru(short int i) {
  return (__half)i;
}

__device__  __half __short2half_rz(short int i) {
  return (__half)i;
}

__device__  __half __uint2half_rd(unsigned int i) {
  return (__half)i;
}

__device__  __half __uint2half_rn(unsigned int i) {
  return (__half)i;
}

__device__  __half __uint2half_ru(unsigned int i) {
  return (__half)i;
}

__device__  __half __uint2half_rz(unsigned int i) {
  return (__half)i;
}

__device__  __half __ull2half_rd(unsigned long long int i) {
  return (__half)i;
}

__device__  __half __ull2half_rn(unsigned long long int i) {
  return (__half)i;
}

__device__  __half __ull2half_ru(unsigned long long int i) {
  return (__half)i;
}

__device__  __half __ull2half_rz(unsigned long long int i) {
  return (__half)i;
}

__device__  __half __ushort2half_rd(unsigned short int i) {
  return (__half)i;
}

__device__  __half __ushort2half_rn(unsigned short int i) {
  return (__half)i;
}

__device__  __half __ushort2half_ru(unsigned short int i) {
  return (__half)i;
}

__device__  __half __ushort2half_rz(unsigned short int i) {
  return (__half)i;
}

__device__  __half __ushort_as_half(const unsigned short int i) {
  hipHalfHolder hH;
  hH.s = i;
  return hH.h;
}


/*
Soft Implementation. Use it for backup.
*/


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


