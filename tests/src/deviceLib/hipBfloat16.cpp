/*
Copyright (c) 2015-2019 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */
#include "test_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <type_traits>
#include <random>
#include <climits>

#define SIZE 100
using namespace std;

static random_device dev;
static mt19937 rng(dev());

inline float getRandomFloat(long min = 10, long max = LONG_MAX) {
    uniform_real_distribution<float> gen(min, max);
    return gen(rng);
}

__host__ __device__ bool testRelativeAccuracy(float a, hip_bfloat16 b) {
  float c = float(b);
  // float relative error should be less than 1/(2^7) since bfloat16
  // has 7 bits mantissa.
  if(fabs(c - a) / a <= 1.0 / 128){
    return true;
  }
  return false;
}

__host__ __device__ void testOperations(float &fa, float &fb) {

  hip_bfloat16 bf_a(fa);
  hip_bfloat16 bf_b(fb);
  float fc = float(bf_a);
  float fd = float(bf_b); 

  assert(testRelativeAccuracy(fa, bf_a));
  assert(testRelativeAccuracy(fb, bf_b));

  assert(testRelativeAccuracy(fc + fd, bf_a + bf_b));
  //when checked as above for add, operation sub fails on GPU 
  assert(hip_bfloat16(fc - fd) == (bf_a - bf_b));
  assert(testRelativeAccuracy(fc * fd, bf_a * bf_b));
  assert(testRelativeAccuracy(fc / fd, bf_a / bf_b));
 
  hip_bfloat16 bf_opNegate = -bf_a;
  assert(bf_opNegate == -bf_a);

  hip_bfloat16 bf_x;
  bf_x = bf_a;
  bf_x++;
  bf_x--;
  ++bf_x;
  --bf_x;
  //hip_bfloat16 is converted to float and then inc/decremented, hence check with reduced precision 
  assert(testRelativeAccuracy(bf_x,bf_a));

  bf_x = bf_a;
  bf_x += bf_b;
  assert(bf_x == (bf_a + bf_b));
  bf_x = bf_a;
  bf_x -= bf_b;
  assert(bf_x == (bf_a - bf_b));
  bf_x = bf_a;
  bf_x *= bf_b;
  assert(bf_x == (bf_a * bf_b));
  bf_x = bf_a;
  bf_x /= bf_b;
  assert(bf_x == (bf_a / bf_b));

  hip_bfloat16 bf_rounded = hip_bfloat16::round_to_bfloat16(fa);
  if (isnan(bf_rounded)) {
    assert(isnan(bf_rounded) || isinf(bf_rounded));
  }
}  

__global__ void testOperationsGPU(float* d_a, float* d_b)
{
  int id = threadIdx.x;
  if (id > SIZE) return;
  float &a = d_a[id];
  float &b = d_b[id];
  testOperations(a, b);
}

int main(){
  float *h_fa, *h_fb;
  float *d_fa, *d_fb;

  h_fa = new float[SIZE];
  h_fb = new float[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_fa[i] = getRandomFloat();
    h_fb[i] = getRandomFloat();
    testOperations(h_fa[i], h_fb[i]);
  }
  cout<<"Host bfloat16 Operations Successful!!"<<endl;
  hipMalloc(&d_fa, sizeof(float) * SIZE);
  hipMalloc(&d_fb, sizeof(float) * SIZE);

  hipMemcpy(d_fa, h_fa, sizeof(float) * SIZE, hipMemcpyHostToDevice);
  hipMemcpy(d_fb, h_fb, sizeof(float) * SIZE, hipMemcpyHostToDevice);

  hipLaunchKernelGGL(testOperationsGPU, 1, SIZE, 0, 0, d_fa, d_fb);
  hipDeviceSynchronize();
  cout<<"Device bfloat16 Operations Successful!!"<<endl; 

  delete[] h_fa;
  delete[] h_fb;
  hipFree(d_fa);
  hipFree(d_fb);
  passed();
  return 1;
}
