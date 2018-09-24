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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include "test_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#define HIP_ASSERT(status) assert(status == hipSuccess)

#define LEN 50
#define SIZE (LEN * sizeof(bool))

  __global__ void kernelTestFMA(bool *Ad) {
    float f = 1.0f / 3.0f;
    double d = f;
    int i = 0;
    auto Check = [&](bool Cond) { Ad[i++] = Cond; };
    // f * f + 3.0f will be different if promoted to double.
    float floatResult = fma(f, f, 3.0f);
    double doubleResult = fma(d, d, 3.0);
    Check(floatResult != doubleResult);

    // check promote to float.
    Check(fma(f, f, 3) == floatResult);
    Check(fma(f, f, (char)3) == floatResult);
    Check(fma(f, f, (unsigned char)3) == floatResult);
    Check(fma(f, f, (short)3) == floatResult);
    Check(fma(f, f, (unsigned short)3) == floatResult);
    Check(fma(f, f, (int)3) == floatResult);
    Check(fma(f, f, (unsigned int)3) == floatResult);
    Check(fma(f, f, (long)3) == floatResult);
    Check(fma(f, f, (unsigned long)3) == floatResult);
    Check(fma(f, f, true) == fma(f, f, 1.0f));

    // check promote to double.
    Check(fma(d, (double)f, 3) == doubleResult);
    Check(fma(d, (double)f, (char)3) == doubleResult);
    Check(fma(d, (double)f, (unsigned char)3) == doubleResult);
    Check(fma(d, (double)f, (short)3) == doubleResult);
    Check(fma(d, (double)f, (unsigned short)3) == doubleResult);
    Check(fma(d, (double)f, (int)3) == doubleResult);
    Check(fma(d, (double)f, (unsigned int)3) == doubleResult);
    Check(fma(d, (double)f, (long)3) == doubleResult);
    Check(fma(d, (double)f, (unsigned long)3) == doubleResult);
    Check(fma(d, (double)f, true) == fma((double)f, (double)f, 1.0));

    while (i < LEN)
      Check(true);
  }

  void runTestFMA() {
    bool *Ad;
    bool A[LEN];
    for (unsigned i = 0; i < LEN; i++) {
      A[i] = 0;
    }

    HIP_ASSERT(hipMalloc((void **)&Ad, SIZE));
    hipLaunchKernelGGL(kernelTestFMA, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Ad);
    HIP_ASSERT(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

    for (unsigned i = 0; i < LEN; i++) {
      assert(A[i]);
    }
  }

    __global__ void kernelTestHalfFMA(bool *Ad) {
      _Float16 h = (_Float16)(1.0f/3.0f);
      float f = h;
      double d = f;
      int i = 0;
      auto Check = [&](bool Cond) { Ad[i++] = Cond; };
      // h * h + 3 will be different if promoted to float.
      _Float16 halfResult = fma(h, h, (_Float16)3);
      float floatResult = fma(f, f, 3.0f);
      double doubleResult = fma(d, d, 3.0);
      Check(halfResult != floatResult);
      Check(halfResult != doubleResult);

      // check promote to half.
      Check(fma(h, h, 3) == halfResult);
      Check(fma(h, h, (char)3) == halfResult);
      Check(fma(h, h, (unsigned char)3) == halfResult);
      Check(fma(h, h, (short)3) == halfResult);
      Check(fma(h, h, (unsigned short)3) == halfResult);
      Check(fma(h, h, (int)3) == halfResult);
      Check(fma(h, h, (unsigned int)3) == halfResult);
      Check(fma(h, h, (long)3) == halfResult);
      Check(fma(h, h, (unsigned long)3) == halfResult);
      Check(fma(h, h, true) == fma(h, h, (_Float16)1));

      while (i < LEN)
        Check(true);
    }

  void runTestHalfFMA() {
    bool *Ad;
    bool A[LEN];
    for (unsigned i = 0; i < LEN; i++) {
      A[i] = 0;
    }

    HIP_ASSERT(hipMalloc((void **)&Ad, SIZE));
    hipLaunchKernelGGL(kernelTestHalfFMA, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Ad);
    HIP_ASSERT(hipMemcpy(A, Ad, SIZE, hipMemcpyDeviceToHost));

    for (unsigned i = 0; i < LEN; i++) {
      assert(A[i]);
    }
  }

int main() {
  runTestFMA();
  runTestHalfFMA();
  passed();
}
