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

#include<iostream>
#include<assert.h>
#include <hip/hip_vector_types.h>

#define cmpFloat1(in, exp) \
  if(in.x != exp) { \
    std::cout<<"Failed at: "<<__LINE__<<" in func: "<<__func__<<" expected output: "<<exp<<" but got: "<<in.x<<std::endl; \
    assert(-1); \
  } \

#define cmpFloat2(in, exp) \
  if(in.x != exp || in.y != exp) { \
    std::cout<<"Failed at: "<<__LINE__<<" in func: "<<__func__<<" expected output: "<<exp<<" but got: "<<in.x<<","<<in.y<<std::endl; \
    assert(-1); \
  } \

#define cmpFloat3(in, exp) \
  if(in.x != exp || in.y != exp || in.z != exp) { \
    std::cout<<"Failed at: "<<__LINE__<<" in func: "<<__func__<<" expected output: "<<exp<<" but got: "<<in.x<<","<<in.y<<","<<in.z<<std::endl; \
    assert(-1); \
  } \

#define cmpFloat4(in, exp) \
  if(in.x != exp || in.y != exp || in.z != exp || in.w != exp ) { \
    std::cout<<"Failed at: "<<__LINE__<<" in func: "<<__func__<<" expected output: "<<exp<<" but got: "<<in.x<<","<<in.y<<","<<in.z<<","<<in.w<<std::endl; \
    assert(-1); \
  } \

bool TestUChar1() {
  uchar1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, 253);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUChar2() {
  uchar2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, 253);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUChar3() {
  uchar3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, 253);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUChar4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, 253);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestChar1() {
  char1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, (char)253);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestChar2() {
  char2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, (char)253);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestChar3() {
  char3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, (char)253);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestChar4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, (char)253);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUShort1() {
  ushort1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, (unsigned short)65533);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUShort2() {
  ushort2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, (unsigned short)65533);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUShort3() {
  ushort3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, (unsigned short)65533);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUShort4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, (unsigned short)65533);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestShort1() {
  short1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, (signed short)65533);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestShort2() {
  short2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, (signed short)65533);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestShort3() {
  short3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, (signed short)65533);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestShort4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, (signed short)65533);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}


bool TestUInt1() {
  uint1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, (unsigned int)4294967293);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUInt2() {
  uint2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, (unsigned int)4294967293);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUInt3() {
  uint3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, (unsigned int)4294967293);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestUInt4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, (unsigned int)4294967293);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestInt1() {
  int1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, (signed int)4294967293);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestInt2() {
  int2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, (signed int)4294967293);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestInt3() {
  int3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, (signed int)4294967293);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestInt4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, (signed int)4294967293);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestULong1() {
  ulong1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, 18446744073709551613UL);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestULong2() {
  ulong2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, 18446744073709551613UL);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestULong3() {
  ulong3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, 18446744073709551613UL);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestULong4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, 18446744073709551613UL);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestLong1() {
  long1 f1, f2, f3;
  f1.x = 1;
  f2.x = 1;
  f3 = f1 + f2;
  cmpFloat1(f3, 2);
  f2 = f3 - f1;
  cmpFloat1(f2, 1);
  f1 = f2 * f3;
  cmpFloat1(f1, 2);
  f2 = f1 / f3;
  cmpFloat1(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat1(f3, 0);
  f1 = f3 & f2;
  cmpFloat1(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat1(f2, 0);
  f1.x = 1;
  f2.x = 2;
  f3 = f1 << f2;
  cmpFloat1(f3, 4);
  f2 = f3 >> f1;
  cmpFloat1(f2, 2);

  f1.x = 2;
  f2.x = 1;
  f1 += f2;
  cmpFloat1(f1, 3);
  f1 -= f2;
  cmpFloat1(f1, 2);
  f1 *= f2;
  cmpFloat1(f1, 2);
  f1 /= f2;
  cmpFloat1(f1, 2);
  f1 %= f2;
  cmpFloat1(f1, 0);
  f1 &= f2;
  cmpFloat1(f1, 0);
  f1 |= f2;
  cmpFloat1(f1, 1);
  f1 ^= f2;
  cmpFloat1(f1, 0);
  f1.x = 1;
  f1 <<= f2;
  cmpFloat1(f1, 2);
  f1 >>= f2;
  cmpFloat1(f1, 1);

  f1.x = 2;
  f2 = f1++;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 2);
  f2 = f1--;
  cmpFloat1(f2, 3);
  cmpFloat1(f1, 2);
  f2 = ++f1;
  cmpFloat1(f1, 3);
  cmpFloat1(f2, 3);
  f2 = --f1;
  cmpFloat1(f1, 2);
  cmpFloat1(f2, 2);

  f2 = ~f1;
  cmpFloat1(f2, -3);
  assert(!f1 == false);

  f1.x = 3;
  f2.x = 4;
  f3.x = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestLong2() {
  long2 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f2.x = 1;
  f2.y = 1;
  f3 = f1 + f2;
  cmpFloat2(f3, 2);
  f2 = f3 - f1;
  cmpFloat2(f2, 1);
  f1 = f2 * f3;
  cmpFloat2(f1, 2);
  f2 = f1 / f3;
  cmpFloat2(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat2(f3, 0);
  f1 = f3 & f2;
  cmpFloat2(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat2(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f2.x = 2;
  f2.y = 2;
  f3 = f1 << f2;
  cmpFloat2(f3, 4);
  f2 = f3 >> f1;
  cmpFloat2(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f2.x = 1;
  f2.y = 1;
  f1 += f2;
  cmpFloat2(f1, 3);
  f1 -= f2;
  cmpFloat2(f1, 2);
  f1 *= f2;
  cmpFloat2(f1, 2);
  f1 /= f2;
  cmpFloat2(f1, 2);
  f1 %= f2;
  cmpFloat2(f1, 0);
  f1 &= f2;
  cmpFloat2(f1, 0);
  f1 |= f2;
  cmpFloat2(f1, 1);
  f1 ^= f2;
  cmpFloat2(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1 <<= f2;
  cmpFloat2(f1, 2);
  f1 >>= f2;
  cmpFloat2(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f2 = f1++;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 2);
  f2 = f1--;
  cmpFloat2(f2, 3);
  cmpFloat2(f1, 2);
  f2 = ++f1;
  cmpFloat2(f1, 3);
  cmpFloat2(f2, 3);
  f2 = --f1;
  cmpFloat2(f1, 2);
  cmpFloat2(f2, 2);

  f2 = ~f1;
  cmpFloat2(f2, -3);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f2.x = 4;
  f2.y = 4;
  f3.x = 3;
  f3.y = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestLong3() {
  long3 f1, f2, f3;
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f3 = f1 + f2;
  cmpFloat3(f3, 2);
  f2 = f3 - f1;
  cmpFloat3(f2, 1);
  f1 = f2 * f3;
  cmpFloat3(f1, 2);
  f2 = f1 / f3;
  cmpFloat3(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat3(f3, 0);
  f1 = f3 & f2;
  cmpFloat3(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat3(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f3 = f1 << f2;
  cmpFloat3(f3, 4);
  f2 = f3 >> f1;
  cmpFloat3(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f1 += f2;
  cmpFloat3(f1, 3);
  f1 -= f2;
  cmpFloat3(f1, 2);
  f1 *= f2;
  cmpFloat3(f1, 2);
  f1 /= f2;
  cmpFloat3(f1, 2);
  f1 %= f2;
  cmpFloat3(f1, 0);
  f1 &= f2;
  cmpFloat3(f1, 0);
  f1 |= f2;
  cmpFloat3(f1, 1);
  f1 ^= f2;
  cmpFloat3(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1 <<= f2;
  cmpFloat3(f1, 2);
  f1 >>= f2;
  cmpFloat3(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f2 = f1++;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 2);
  f2 = f1--;
  cmpFloat3(f2, 3);
  cmpFloat3(f1, 2);
  f2 = ++f1;
  cmpFloat3(f1, 3);
  cmpFloat3(f2, 3);
  f2 = --f1;
  cmpFloat3(f1, 2);
  cmpFloat3(f2, 2);

  f2 = ~f1;
  cmpFloat3(f2, -3);
  assert(!f1 == false);

  f1.x = 3;
  f1.y = 3;
  f1.z = 3;
  f2.x = 4;
  f2.y = 4;
  f2.z = 4;
  f3.x = 3;
  f3.y = 3;
  f3.z = 3;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}

bool TestLong4() {
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
  cmpFloat4(f3, 2);
  f2 = f3 - f1;
  cmpFloat4(f2, 1);
  f1 = f2 * f3;
  cmpFloat4(f1, 2);
  f2 = f1 / f3;
  cmpFloat4(f2, 2/2);
  f3 = f1 % f2;
  cmpFloat4(f3, 0);
  f1 = f3 & f2;
  cmpFloat4(f1, 0);
  f2 = f1 ^ f3;
  cmpFloat4(f2, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f2.x = 2;
  f2.y = 2;
  f2.z = 2;
  f2.w = 2;
  f3 = f1 << f2;
  cmpFloat4(f3, 4);
  f2 = f3 >> f1;
  cmpFloat4(f2, 2);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2.x = 1;
  f2.y = 1;
  f2.z = 1;
  f2.w = 1;
  f1 += f2;
  cmpFloat4(f1, 3);
  f1 -= f2;
  cmpFloat4(f1, 2);
  f1 *= f2;
  cmpFloat4(f1, 2);
  f1 /= f2;
  cmpFloat4(f1, 2);
  f1 %= f2;
  cmpFloat4(f1, 0);
  f1 &= f2;
  cmpFloat4(f1, 0);
  f1 |= f2;
  cmpFloat4(f1, 1);
  f1 ^= f2;
  cmpFloat4(f1, 0);
  f1.x = 1;
  f1.y = 1;
  f1.z = 1;
  f1.w = 1;
  f1 <<= f2;
  cmpFloat4(f1, 2);
  f1 >>= f2;
  cmpFloat4(f1, 1);

  f1.x = 2;
  f1.y = 2;
  f1.z = 2;
  f1.w = 2;
  f2 = f1++;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 2);
  f2 = f1--;
  cmpFloat4(f2, 3);
  cmpFloat4(f1, 2);
  f2 = ++f1;
  cmpFloat4(f1, 3);
  cmpFloat4(f2, 3);
  f2 = --f1;
  cmpFloat4(f1, 2);
  cmpFloat4(f2, 2);

  f2 = ~f1;
  cmpFloat4(f2, -3);
  assert(!f1 == false);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  assert((f1 && f2) == true);
  assert((f1 || f2) == true);
  return true;
}


bool TestFloat1() {
  float1 f1, f2, f3;
//  float1 f4(1);
//  cmpFloat1(f4, 1.0f);
//  float1 f5(2.0f);
//  cmpFloat1(f5, 2.0f);
  f1.x = 1.0f;
  f2.x = 1.0f;
  f3 = f1 + f2;
  cmpFloat1(f3, 2.0f);
  f2 = f3 - f1;
  cmpFloat1(f2, 1.0f);
  f1 = f2 * f3;
  cmpFloat1(f1, 2.0f);
  f2 = f1 / f3;
  cmpFloat1(f2, 2.0f/2.0f);
  f1 += f2;
  cmpFloat1(f1, 3.0f);
  f1 -= f2;
  cmpFloat1(f1, 2.0f);
  f1 *= f2;
  cmpFloat1(f1, 2.0f);
  f1 /= f2;
  cmpFloat1(f1, 2.0f);
  f2 = f1++;
  cmpFloat1(f1, 3.0f);
  cmpFloat1(f2, 2.0f);
  f2 = f1--;
  cmpFloat1(f2, 3.0f);
  cmpFloat1(f1, 2.0f);
  f2 = ++f1;
  cmpFloat1(f1, 3.0f);
  cmpFloat1(f2, 3.0f);
  f2 = --f1;
  cmpFloat1(f1, 2.0f);
  cmpFloat1(f1, 2.0f);

  f1.x = 3.0f;
  f2.x = 4.0f;
  f3.x = 3.0f;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  return true;
}

bool TestFloat2() {
  float2 f1, f2, f3;
  f1.x = 1.0f;
  f1.y = 1.0f;
  f2.x = 1.0f;
  f2.y = 1.0f;
  f3 = f1 + f2;
  cmpFloat2(f3, 2.0f);
  f2 = f3 - f1;
  cmpFloat2(f2, 1.0f);
  f1 = f2 * f3;
  cmpFloat2(f1, 2.0f);
  f2 = f1 / f3;
  cmpFloat2(f2, 2.0f/2.0f);
  f1 += f2;
  cmpFloat2(f1, 3.0f);
  f1 -= f2;
  cmpFloat2(f1, 2.0f);
  f1 *= f2;
  cmpFloat2(f1, 2.0f);
  f1 /= f2;
  cmpFloat2(f1, 2.0f);

  f2 = f1++;
  cmpFloat2(f1, 3.0f);
  cmpFloat2(f2, 2.0f);
  f2 = f1--;
  cmpFloat2(f2, 3.0f);
  cmpFloat2(f1, 2.0f);
  f2 = ++f1;
  cmpFloat2(f1, 3.0f);
  cmpFloat2(f2, 3.0f);
  f2 = --f1;
  cmpFloat2(f1, 2.0f);
  cmpFloat2(f1, 2.0f);

  f1.x = 3.0f;
  f1.y = 3.0f;
  f2.x = 4.0f;
  f2.y = 4.0f;
  f3.x = 3.0f;
  f3.y = 3.0f;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);


  return true;
}

bool TestFloat3() {
  float3 f1, f2, f3;
  f1.x = 1.0f;
  f1.y = 1.0f;
  f1.z = 1.0f;
  f2.x = 1.0f;
  f2.y = 1.0f;
  f2.z = 1.0f;
  f3 = f1 + f2;
  cmpFloat3(f3, 2.0f);
  f2 = f3 - f1;
  cmpFloat3(f2, 1.0f);
  f1 = f2 * f3;
  cmpFloat3(f1, 2.0f);
  f2 = f1 / f3;
  cmpFloat3(f2, 2.0f/2.0f);
  f1 += f2;
  cmpFloat3(f1, 3.0f);
  f1 -= f2;
  cmpFloat3(f1, 2.0f);
  f1 *= f2;
  cmpFloat3(f1, 2.0f);
  f1 /= f2;
  f2 = f1++;
  cmpFloat3(f1, 3.0f);
  cmpFloat3(f2, 2.0f);
  f2 = f1--;
  cmpFloat3(f2, 3.0f);
  cmpFloat3(f1, 2.0f);
  f2 = ++f1;
  cmpFloat3(f1, 3.0f);
  cmpFloat3(f2, 3.0f);
  f2 = --f1;
  cmpFloat3(f1, 2.0f);
  cmpFloat3(f1, 2.0f);

  f1.x = 3.0f;
  f1.y = 3.0f;
  f1.z = 3.0f;
  f2.x = 4.0f;
  f2.y = 4.0f;
  f2.z = 4.0f;
  f3.x = 3.0f;
  f3.y = 3.0f;
  f3.z = 3.0f;
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);


  return true;
}


bool TestFloat4() {
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
  cmpFloat4(f3, 2.0f);
  f2 = f3 - f1;
  cmpFloat4(f2, 1.0f);
  f1 = f2 * f3;
  cmpFloat4(f1, 2.0f);
  f2 = f1 / f3;
  cmpFloat4(f2, 2.0f/2.0f);
  f1 += f2;
  cmpFloat4(f1, 3.0f);
  f1 -= f2;
  cmpFloat4(f1, 2.0f);
  f1 *= f2;
  cmpFloat4(f1, 2.0f);
  f1 /= f2;
  f2 = f1++;
  cmpFloat4(f1, 3.0f);
  cmpFloat4(f2, 2.0f);
  f2 = f1--;
  cmpFloat4(f2, 3.0f);
  cmpFloat4(f1, 2.0f);
  f2 = ++f1;
  cmpFloat4(f1, 3.0f);
  cmpFloat4(f2, 3.0f);
  f2 = --f1;
  cmpFloat4(f1, 2.0f);
  cmpFloat4(f1, 2.0f);

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
  assert((f1 == f2) == false);
  assert((f1 != f2) == true);
  assert((f1 < f2) == true);
  assert((f2 > f1) == true);
  assert((f1 >= f3) == true);
  assert((f1 <= f3) == true);

  return true;
}



int main() {
  assert(sizeof(float1) == 4);
  assert(sizeof(float2) == 8);
  assert(sizeof(float3) == 12);
  assert(sizeof(float4) == 16);
  assert(TestFloat1() && TestFloat2() && TestFloat3() && TestFloat4()
    && TestUChar1() && TestUChar2() && TestUChar3() && TestUChar4()
    && TestChar1() && TestChar2() && TestChar3() && TestChar4()
    && TestUShort1() && TestUShort2() && TestUShort3() && TestUShort4()
    && TestShort1() && TestShort2() && TestShort3() && TestShort4()
    && TestUInt1() && TestUInt2() && TestUInt3() && TestUInt4()
    && TestInt1() && TestInt2() && TestInt3() && TestInt4()
    && TestULong1() && TestULong2() && TestULong3() && TestULong4()
    && TestLong1() && TestLong2() && TestLong3() && TestLong4() == true);

  float1 f1 = make_float1(1.0f);
}
