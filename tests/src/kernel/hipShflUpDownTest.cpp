/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <hip/hip_runtime.h>
#include "test_common.h"

template <typename T>
__global__ void shflDownSum(T* a, int size) {
    T val = a[threadIdx.x];
    for (int i = size / 2; i > 0; i /= 2) {
        val += __shfl_down(val, i, size);
    }
    a[threadIdx.x] = val;
}

template <typename T>
__global__ void shflUpSum(T* a, int size) {
    T val = a[threadIdx.x];
    for (int i = size / 2; i > 0; i /= 2) {
        val += __shfl_up(val, i, size);
    }
    a[threadIdx.x] = val;
}

template <typename T>
__global__ void shflXorSum(T* a, int size) {
  T val = a[threadIdx.x];
  for (int i = size/2; i > 0; i /= 2)
    val += __shfl_xor(val, i, size);
  a[threadIdx.x] = val;
}

void getFactor(int& fact) { fact = 101; }
void getFactor(unsigned int& fact) { fact = static_cast<unsigned int>(INT32_MAX)+1; }
void getFactor(float& fact) { fact = 2.5; }
void getFactor(double& fact) { fact = 2.5; }
void getFactor(long& fact) { fact = 202; }
void getFactor(unsigned long& fact) { fact = static_cast<unsigned long>(__LONG_MAX__)+1; }
void getFactor(long long& fact) { fact = 303; }
void getFactor(unsigned long long& fact) { fact = static_cast<unsigned long long>(__LONG_LONG_MAX__)+1; }

template <typename T>
void runTestShflUp() {
    const int size = 32;
    T a[size];
    T cpuSum = 0;
    T factor; getFactor(factor);
    for (int i = 0; i < size; i++) {
        a[i] = i + factor;
        cpuSum += a[i];
    }
    T* d_a;
    hipMalloc(&d_a, sizeof(T) * size);
    hipMemcpy(d_a, &a, sizeof(T) * size, hipMemcpyDefault);
    hipLaunchKernelGGL(shflUpSum<T>, 1, size, 0, 0, d_a, size);
    hipMemcpy(&a, d_a, sizeof(T) * size, hipMemcpyDefault);
    if (a[size - 1] != cpuSum) {
        hipFree(d_a);
        failed("Shfl Up Sum did not match.");
    }
    hipFree(d_a);
}

template <typename T>
void runTestShflDown() {
    const int size = 32;
    T a[size];
    T cpuSum = 0;
    T factor; getFactor(factor);
    for (int i = 0; i < size; i++) {
        a[i] = i + factor;
        cpuSum += a[i];
    }
    T* d_a;
    hipMalloc(&d_a, sizeof(T) * size);
    hipMemcpy(d_a, &a, sizeof(T) * size, hipMemcpyDefault);
    hipLaunchKernelGGL(shflDownSum<T>, 1, size, 0, 0, d_a, size);
    hipMemcpy(&a, d_a, sizeof(T) * size, hipMemcpyDefault);
    if (a[0] != cpuSum) {
        hipFree(d_a);
        failed("Shfl Down Sum did not match.");
    }
    hipFree(d_a);
}

template <typename T>
void runTestShflXor() {
    const int size = 32;
    T a[size];
    T cpuSum = 0;
    T factor; getFactor(factor);
    for (int i = 0; i < size; i++) {
        a[i] = i + factor;
        cpuSum += a[i];
    }
    T* d_a;
    hipMalloc(&d_a, sizeof(T) * size);
    hipMemcpy(d_a, &a, sizeof(T) * size, hipMemcpyDefault);
    hipLaunchKernelGGL(shflXorSum<T>, 1, size, 0, 0, d_a, size);
    hipMemcpy(&a, d_a, sizeof(T) * size, hipMemcpyDefault);
    if (a[0] != cpuSum) {
        hipFree(d_a);
        failed("Shfl Xor Sum did not match.");
    }
    hipFree(d_a);
}
int main() {
    runTestShflUp<int>();
    runTestShflUp<float>();
    runTestShflUp<double>();
    runTestShflUp<long>();
    runTestShflUp<long long>();
    runTestShflUp<unsigned int>();
    runTestShflUp<unsigned long>();
    runTestShflUp<unsigned long long>();

    runTestShflDown<int>();
    runTestShflDown<float>();
    runTestShflDown<double>();
    runTestShflDown<long>();
    runTestShflDown<long long>();
    runTestShflDown<unsigned int>();
    runTestShflDown<unsigned long>();
    runTestShflDown<unsigned long long>();

    runTestShflXor<int>();
    runTestShflXor<float>();
    runTestShflXor<double>();
    runTestShflXor<long>();
    runTestShflXor<long long>();
    runTestShflXor<unsigned int>();
    runTestShflXor<unsigned long>();
    runTestShflXor<unsigned long long>();
    passed();
}
