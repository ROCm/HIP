/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include "test_common.h"

#define HIP_ASSERT(status) assert(status == hipSuccess)

#define LEN 512
#define SIZE (LEN * sizeof(long long))

  static __global__ void kernel1(long long* Ad) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      Ad[tid] = clock() + clock64() + __clock() + __clock64();
  }

  static __global__ void kernel2(long long* Ad) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      Ad[tid] = clock() + clock64() + __clock() + __clock64() - Ad[tid];
  }

  static __global__ void kernel1_gfx11(long long* Ad) {
#ifdef __HIP_PLATFORM_AMD__
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      Ad[tid] = clock() + wall_clock64() + __clock() + __clock64();
#endif
  }

  static __global__ void kernel2_gfx11(long long* Ad) {
#ifdef __HIP_PLATFORM_AMD__
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      Ad[tid] = clock() + wall_clock64() + __clock() + __clock64() - Ad[tid];
#endif
  }

  void run() {
    long long *A, *Ad;
    A = new long long[LEN];
    for (unsigned i = 0; i < LEN; i++) {
        A[i] = 0;
    }

    auto kernel1_used = IsGfx11() ? kernel1_gfx11 : kernel1;
    auto kernel2_used = IsGfx11() ? kernel2_gfx11 : kernel2;

    HIP_ASSERT(hipMalloc((void**)&Ad, SIZE));

    hipLaunchKernelGGL(kernel1_used, dim3(1, 1, 1),
                       dim3(LEN, 1, 1), 0, 0, Ad);
    hipLaunchKernelGGL(kernel2_used, dim3(1, 1, 1),
                       dim3(LEN, 1, 1), 0, 0, Ad);
    HIP_ASSERT(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

    for (unsigned i = 0; i < LEN; i++) {
        assert(0 != A[i]);
    }
  }

int main() {
  run();
  passed();
}
