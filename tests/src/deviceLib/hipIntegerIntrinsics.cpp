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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */


#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include "test_common.h"

#include <algorithm>

using namespace std;

#pragma GCC diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wunused-variable"

__device__ void integer_intrinsics() {
    __brev((unsigned int)10);
    __brevll((unsigned long long)10);
    __byte_perm((unsigned int)0, (unsigned int)0, 0);
    __clz((int)10);
    __clzll((long long)10);
    __ffs((int)10);
    __ffsll((long long)10);
    __hadd((int)1, (int)3);
    __mul24((int)1, (int)2);
    __mul64hi((long long)1, (long long)2);
    __mulhi((int)1, (int)2);
    __popc((unsigned int)4);
    __popcll((unsigned long long)4);
    int a = min((int)4, (int)5);
    int b = max((int)4, (int)5);
    __rhadd((int)1, (int)2);
    __sad((int)1, (int)2, 0);
    __uhadd((unsigned int)1, (unsigned int)3);
    __umul24((unsigned int)1, (unsigned int)2);
    __umul64hi((unsigned long long)1, (unsigned long long)2);
    __umulhi((unsigned int)1, (unsigned int)2);
    __urhadd((unsigned int)1, (unsigned int)2);
    __usad((unsigned int)1, (unsigned int)2, 0);

    assert(1);
}

__global__ void compileIntegerIntrinsics(int ignored) { integer_intrinsics(); }

int main() {
    hipLaunchKernelGGL(compileIntegerIntrinsics, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, 1);
    passed();
}
