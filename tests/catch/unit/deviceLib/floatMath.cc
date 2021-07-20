/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>

#define LEN 512
#define SIZE LEN << 2

__global__ void floatMath(float* In, float* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Out[tid] = __cosf(In[tid]);
  Out[tid] = __exp10f(Out[tid]);
  Out[tid] = __expf(Out[tid]);
  Out[tid] = __frsqrt_rn(Out[tid]);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  Out[tid] = __fsqrt_rd(Out[tid]);
#endif
  Out[tid] = __fsqrt_rn(Out[tid]);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  Out[tid] = __fsqrt_ru(Out[tid]);
  Out[tid] = __fsqrt_rz(Out[tid]);
#endif
  Out[tid] = __log10f(Out[tid]);
  Out[tid] = __log2f(Out[tid]);
  Out[tid] = __logf(Out[tid]);
  Out[tid] = __powf(2.0f, Out[tid]);
  __sincosf(Out[tid], &In[tid], &Out[tid]);
  Out[tid] = __sinf(Out[tid]);
  Out[tid] = __cosf(Out[tid]);
  Out[tid] = __tanf(Out[tid]);
}

TEST_CASE("Unit_deviceFunctions_CompileTest") {
  float *Ind, *Outd;
  auto res = hipMalloc((void**)&Ind, SIZE);
  REQUIRE(res == hipSuccess);
  res = hipMalloc((void**)&Outd, SIZE);
  REQUIRE(res == hipSuccess);
  hipLaunchKernelGGL(floatMath, dim3(LEN, 1, 1), dim3(1, 1, 1), 0, 0, Ind, Outd);
  res = hipDeviceSynchronize();
  REQUIRE(res == hipSuccess);
  res = hipGetLastError();
  REQUIRE(res == hipSuccess);
  HIP_CHECK(hipFree(Ind));
  HIP_CHECK(hipFree(Outd));
}
