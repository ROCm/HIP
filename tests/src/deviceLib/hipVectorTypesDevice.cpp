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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include<iostream>
#include<hip/hip_vector_types.h>
#include"test_common.h"
#define cmpVal1(in, exp) \
  if(in.x != exp) { \
  } \

#define cmpVal2(in, exp) \
  if(in.x != exp || in.y != exp) { \
  } \

#define cmpVal3(in, exp) \
  if(in.x != exp || in.y != exp || in.z != exp) { \
  } \

#define cmpVal4(in, exp) \
  if(in.x != exp || in.y != exp || in.z != exp || in.w != exp ) { \
  } \

__device__ bool TestUChar1() {
  uchar1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, 253);

  f1.x = 3;
  f1 = f1 * (unsigned char)1;
  cmpVal1(f1, 3);
  f1 = (unsigned char)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (signed char)1;
  cmpVal1(f1, 3);
  f1 = (signed char)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (unsigned short)1;
  cmpVal1(f1, 3);
  f1 = (unsigned short)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (signed short)1;
  cmpVal1(f1, 3);
  f1 = (signed short)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (unsigned int)1;
  cmpVal1(f1, 3);
  f1 = (unsigned int)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (signed int)1;
  cmpVal1(f1, 3);
  f1 = (signed int)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (float)1;
  cmpVal1(f1, 3);
  f1 = (float)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (unsigned long)1;
  cmpVal1(f1, 3);
  f1 = (unsigned long)1 * f1;
  cmpVal1(f1, 3);
  f1 = f1 * (signed long)1;
  cmpVal1(f1, 3);
  f1 = (signed long)1 * f1;
  cmpVal1(f1, 3);

//  signed char sc = 1;

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUChar2() {
  uchar2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, 253);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUChar3() {
  uchar3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, 253);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUChar4() {
  uchar4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, 253);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestChar1() {
  char1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, (char)253);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestChar2() {
  char2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, (char)253);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestChar3() {
  char3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, (char)253);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestChar4() {
  char4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, (char)253);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUShort1() {
  ushort1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, (unsigned short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUShort2() {
  ushort2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, (unsigned short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUShort3() {
  ushort3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, (unsigned short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUShort4() {
  ushort4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, (unsigned short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestShort1() {
  short1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, (signed short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestShort2() {
  short2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, (signed short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestShort3() {
  short3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, (signed short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestShort4() {
  short4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, (signed short)65533);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}


__device__ bool TestUInt1() {
  uint1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, (unsigned int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUInt2() {
  uint2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, (unsigned int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUInt3() {
  uint3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, (unsigned int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestUInt4() {
  uint4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, (unsigned int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestInt1() {
  int1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, (signed int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestInt2() {
  int2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, (signed int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestInt3() {
  int3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, (signed int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestInt4() {
  int4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, (signed int)4294967293);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestULong1() {
  ulong1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, 18446744073709551613UL);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestULong2() {
  ulong2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, 18446744073709551613UL);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestULong3() {
  ulong3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, 18446744073709551613UL);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestULong4() {
  ulong4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, 18446744073709551613UL);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestLong1() {
  long1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestLong2() {
  long2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestLong3() {
  long3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestLong4() {
  long4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}


__device__ bool TestFloat1() {
  float1 f1, f2, f3;
//  float1 f4(1);
//  cmpVal1(f4, 1.0f);
//  float1 f5(2.0f);
//  cmpVal1(f5, 2.0f);
  f1.x = 1.0f;
  f2.x = 1.0f;
  f3 = f1 + f2;
  cmpVal1(f3, 2.0f);
  f2 = f3 - f1;
  cmpVal1(f2, 1.0f);
  f1 = f2 * f3;
  cmpVal1(f1, 2.0f);
  f2 = f1 / f3;
  cmpVal1(f2, 2.0f/2.0f);
  f1 += f2;
  cmpVal1(f1, 3.0f);
  f1 -= f2;
  cmpVal1(f1, 2.0f);
  f1 *= f2;
  cmpVal1(f1, 2.0f);
  f1 /= f2;
  cmpVal1(f1, 2.0f);
  f2 = f1++;
  cmpVal1(f1, 3.0f);
  cmpVal1(f2, 2.0f);
  f2 = f1--;
  cmpVal1(f2, 3.0f);
  cmpVal1(f1, 2.0f);
  f2 = ++f1;
  cmpVal1(f1, 3.0f);
  cmpVal1(f2, 3.0f);
  f2 = --f1;
  cmpVal1(f1, 2.0f);
  cmpVal1(f1, 2.0f);

  f1.x = 3.0f;
  f2.x = 4.0f;
  f3.x = 3.0f;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  return true;
}

__device__ bool TestFloat2() {
  float2 f1, f2, f3;
  f1.x = 1.0f;
  f1.y = 1.0f;
  f2.x = 1.0f;
  f2.y = 1.0f;
  f3 = f1 + f2;
  cmpVal2(f3, 2.0f);
  f2 = f3 - f1;
  cmpVal2(f2, 1.0f);
  f1 = f2 * f3;
  cmpVal2(f1, 2.0f);
  f2 = f1 / f3;
  cmpVal2(f2, 2.0f/2.0f);
  f1 += f2;
  cmpVal2(f1, 3.0f);
  f1 -= f2;
  cmpVal2(f1, 2.0f);
  f1 *= f2;
  cmpVal2(f1, 2.0f);
  f1 /= f2;
  cmpVal2(f1, 2.0f);

  f2 = f1++;
  cmpVal2(f1, 3.0f);
  cmpVal2(f2, 2.0f);
  f2 = f1--;
  cmpVal2(f2, 3.0f);
  cmpVal2(f1, 2.0f);
  f2 = ++f1;
  cmpVal2(f1, 3.0f);
  cmpVal2(f2, 3.0f);
  f2 = --f1;
  cmpVal2(f1, 2.0f);
  cmpVal2(f1, 2.0f);

  f1.x = 3.0f;
  f1.y = 3.0f;
  f2.x = 4.0f;
  f2.y = 4.0f;
  f3.x = 3.0f;
  f3.y = 3.0f;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}


  return true;
}

__device__ bool TestFloat3() {
  float3 f1, f2, f3;
  f1.x = 1.0f;
  f1.y = 1.0f;
  f1.z = 1.0f;
  f2.x = 1.0f;
  f2.y = 1.0f;
  f2.z = 1.0f;
  f3 = f1 + f2;
  cmpVal3(f3, 2.0f);
  f2 = f3 - f1;
  cmpVal3(f2, 1.0f);
  f1 = f2 * f3;
  cmpVal3(f1, 2.0f);
  f2 = f1 / f3;
  cmpVal3(f2, 2.0f/2.0f);
  f1 += f2;
  cmpVal3(f1, 3.0f);
  f1 -= f2;
  cmpVal3(f1, 2.0f);
  f1 *= f2;
  cmpVal3(f1, 2.0f);
  f1 /= f2;
  f2 = f1++;
  cmpVal3(f1, 3.0f);
  cmpVal3(f2, 2.0f);
  f2 = f1--;
  cmpVal3(f2, 3.0f);
  cmpVal3(f1, 2.0f);
  f2 = ++f1;
  cmpVal3(f1, 3.0f);
  cmpVal3(f2, 3.0f);
  f2 = --f1;
  cmpVal3(f1, 2.0f);
  cmpVal3(f1, 2.0f);

  f1.x = 3.0f;
  f1.y = 3.0f;
  f1.z = 3.0f;
  f2.x = 4.0f;
  f2.y = 4.0f;
  f2.z = 4.0f;
  f3.x = 3.0f;
  f3.y = 3.0f;
  f3.z = 3.0f;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}


  return true;
}


__device__ bool TestFloat4() {
  float4 f1, f2, f3;
  f1.x = 1.0f;
  f1.y = 1.0f;
  f1.z = 1.0f;
  f1.w = 1.0f;
  f2.x = 1.0f;
  f2.y = 1.0f;
  f2.z = 1.0f;
  f2.w = 1.0f;
  f3 = f1 + f2;
  cmpVal4(f3, 2.0f);
  f2 = f3 - f1;
  cmpVal4(f2, 1.0f);
  f1 = f2 * f3;
  cmpVal4(f1, 2.0f);
  f2 = f1 / f3;
  cmpVal4(f2, 2.0f/2.0f);
  f1 += f2;
  cmpVal4(f1, 3.0f);
  f1 -= f2;
  cmpVal4(f1, 2.0f);
  f1 *= f2;
  cmpVal4(f1, 2.0f);
  f1 /= f2;
  f2 = f1++;
  cmpVal4(f1, 3.0f);
  cmpVal4(f2, 2.0f);
  f2 = f1--;
  cmpVal4(f2, 3.0f);
  cmpVal4(f1, 2.0f);
  f2 = ++f1;
  cmpVal4(f1, 3.0f);
  cmpVal4(f2, 3.0f);
  f2 = --f1;
  cmpVal4(f1, 2.0f);
  cmpVal4(f1, 2.0f);

  f1.x = 3.0f;
  f1.y = 3.0f;
  f1.z = 3.0f;
  f1.w = 3.0f;
  f2.x = 4.0f;
  f2.y = 4.0f;
  f2.z = 4.0f;
  f2.w = 4.0f;
  f3.x = 3.0f;
  f3.y = 3.0f;
  f3.z = 3.0f;
  f3.w = 3.0f;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  return true;
}

__device__ bool TestULongLong1() {
  ulonglong1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpVal1(f3, 2);
  f2 = f3 - f1;
  cmpVal1(f2, 1);
  f1 = f2 * f3;
  cmpVal1(f1, 2);
  f2 = f1 / f3;
  cmpVal1(f2, 2/2);
  f3 = f1 % f2;
  cmpVal1(f3, 0);
  f1 = f3 & f2;
  cmpVal1(f1, 0);
  f2 = f1 ^ f3;
  cmpVal1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpVal1(f3, 4);
  f2 = f3 >> f1;
  cmpVal1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpVal1(f1, 3);
  f1 -= f2;
  cmpVal1(f1, 2);
  f1 *= f2;
  cmpVal1(f1, 2);
  f1 /= f2;
  cmpVal1(f1, 2);
  f1 %= f2;
  cmpVal1(f1, 0);
  f1 &= f2;
  cmpVal1(f1, 0);
  f1 |= f2;
  cmpVal1(f1, 1);
  f1 ^= f2;
  cmpVal1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpVal1(f1, 2);
  f1 >>= f2;
  cmpVal1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpVal1(f1, 3);
  cmpVal1(f2, 2);
  f2 = f1--;
  cmpVal1(f2, 3);
  cmpVal1(f1, 2);
  f2 = ++f1;
  cmpVal1(f1, 3);
  cmpVal1(f2, 3);
  f2 = --f1;
  cmpVal1(f1, 2);
  cmpVal1(f2, 2);

  f2 = ~f1;
  cmpVal1(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}


__device__ bool TestULongLong2() {
  ulonglong2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpVal2(f3, 2);
  f2 = f3 - f1;
  cmpVal2(f2, 1);
  f1 = f2 * f3;
  cmpVal2(f1, 2);
  f2 = f1 / f3;
  cmpVal2(f2, 2/2);
  f3 = f1 % f2;
  cmpVal2(f3, 0);
  f1 = f3 & f2;
  cmpVal2(f1, 0);
  f2 = f1 ^ f3;
  cmpVal2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpVal2(f3, 4);
  f2 = f3 >> f1;
  cmpVal2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpVal2(f1, 3);
  f1 -= f2;
  cmpVal2(f1, 2);
  f1 *= f2;
  cmpVal2(f1, 2);
  f1 /= f2;
  cmpVal2(f1, 2);
  f1 %= f2;
  cmpVal2(f1, 0);
  f1 &= f2;
  cmpVal2(f1, 0);
  f1 |= f2;
  cmpVal2(f1, 1);
  f1 ^= f2;
  cmpVal2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpVal2(f1, 2);
  f1 >>= f2;
  cmpVal2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpVal2(f1, 3);
  cmpVal2(f2, 2);
  f2 = f1--;
  cmpVal2(f2, 3);
  cmpVal2(f1, 2);
  f2 = ++f1;
  cmpVal2(f1, 3);
  cmpVal2(f2, 3);
  f2 = --f1;
  cmpVal2(f1, 2);
  cmpVal2(f2, 2);

  f2 = ~f1;
  cmpVal2(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestULongLong3() {
  ulonglong3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpVal3(f3, 2);
  f2 = f3 - f1;
  cmpVal3(f2, 1);
  f1 = f2 * f3;
  cmpVal3(f1, 2);
  f2 = f1 / f3;
  cmpVal3(f2, 2/2);
  f3 = f1 % f2;
  cmpVal3(f3, 0);
  f1 = f3 & f2;
  cmpVal3(f1, 0);
  f2 = f1 ^ f3;
  cmpVal3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpVal3(f3, 4);
  f2 = f3 >> f1;
  cmpVal3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpVal3(f1, 3);
  f1 -= f2;
  cmpVal3(f1, 2);
  f1 *= f2;
  cmpVal3(f1, 2);
  f1 /= f2;
  cmpVal3(f1, 2);
  f1 %= f2;
  cmpVal3(f1, 0);
  f1 &= f2;
  cmpVal3(f1, 0);
  f1 |= f2;
  cmpVal3(f1, 1);
  f1 ^= f2;
  cmpVal3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpVal3(f1, 2);
  f1 >>= f2;
  cmpVal3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpVal3(f1, 3);
  cmpVal3(f2, 2);
  f2 = f1--;
  cmpVal3(f2, 3);
  cmpVal3(f1, 2);
  f2 = ++f1;
  cmpVal3(f1, 3);
  cmpVal3(f2, 3);
  f2 = --f1;
  cmpVal3(f1, 2);
  cmpVal3(f2, 2);

  f2 = ~f1;
  cmpVal3(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}

__device__ bool TestULongLong4() {
  ulonglong4 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f3 = f1 + f2;
  cmpVal4(f3, 2);
  f2 = f3 - f1;
  cmpVal4(f2, 1);
  f1 = f2 * f3;
  cmpVal4(f1, 2);
  f2 = f1 / f3;
  cmpVal4(f2, 2/2);
  f3 = f1 % f2;
  cmpVal4(f3, 0);
  f1 = f3 & f2;
  cmpVal4(f1, 0);
  f2 = f1 ^ f3;
  cmpVal4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpVal4(f3, 4);
  f2 = f3 >> f1;
  cmpVal4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpVal4(f1, 3);
  f1 -= f2;
  cmpVal4(f1, 2);
  f1 *= f2;
  cmpVal4(f1, 2);
  f1 /= f2;
  cmpVal4(f1, 2);
  f1 %= f2;
  cmpVal4(f1, 0);
  f1 &= f2;
  cmpVal4(f1, 0);
  f1 |= f2;
  cmpVal4(f1, 1);
  f1 ^= f2;
  cmpVal4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpVal4(f1, 2);
  f1 >>= f2;
  cmpVal4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpVal4(f1, 3);
  cmpVal4(f2, 2);
  f2 = f1--;
  cmpVal4(f2, 3);
  cmpVal4(f1, 2);
  f2 = ++f1;
  cmpVal4(f1, 3);
  cmpVal4(f2, 3);
  f2 = --f1;
  cmpVal4(f1, 2);
  cmpVal4(f2, 2);

  f2 = ~f1;
  cmpVal4(f2, -3);
  if(!f1 == false){}

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f1.w = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f2.w = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  f3.w = 3;
  if((f1 == f2) == false){}
  if((f1 != f2) == true){}
  if((f1 < f2) == true){}
  if((f2 > f1) == true){}
  if((f1 >= f3) == true){}
  if((f1 <= f3) == true){}

  if((f1 && f2) == true){}
  if((f1 || f2) == true){}
  return true;
}


__global__ void CheckVectorTypes(hipLaunchParm lp, bool *ptr){
  if(TestFloat1() && TestFloat2() && TestFloat3() && TestFloat4()
    && TestUChar1() && TestUChar2() && TestUChar3() && TestUChar4()
    && TestChar1() && TestChar2() && TestChar3() && TestChar4()
    && TestUShort1() && TestUShort2() && TestUShort3() && TestUShort4()
    && TestShort1() && TestShort2() && TestShort3() && TestShort4()
    && TestUInt1() && TestUInt2() && TestUInt3() && TestUInt4()
    && TestInt1() && TestInt2() && TestInt3() && TestInt4()
    && TestULong1() && TestULong2() && TestULong3() && TestULong4()
    && TestLong1() && TestLong2() && TestLong3() && TestLong4()
    && TestULongLong1() && TestULongLong2() && TestULongLong3() && TestULongLong4() == true){
      ptr[0] = true;
    }
}

int main() {
  assert(sizeof(float1) == 4);
  assert(sizeof(float2) == 8);
  assert(sizeof(float3) == 12);
  assert(sizeof(float4) == 16);
 
  bool* ptr = nullptr;
  if (hipMalloc(&ptr, sizeof(bool)) != HIP_SUCCESS) return EXIT_FAILURE;
  std::unique_ptr<bool, decltype(hipFree)*> correct{ptr, hipFree};
  hipLaunchKernel(
      CheckVectorTypes, dim3(1,1,1), dim3(1,1,1), 0, 0, correct.get());
  bool passed = false;
  if (hipMemcpyDtoH(&passed, correct.get(), sizeof(bool)) != HIP_SUCCESS) {
      return EXIT_FAILURE;
  }
 
  if (passed == true){
      std::cout << "PASSED" << std::endl;
      return 0;
  } 
  else 
      return EXIT_FAILURE;
}

