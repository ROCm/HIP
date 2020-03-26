/*
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */


#include "test_common.h"

int main() {
    int* A;
    int* Ad;
    int* Bd;

    // Allocation
    HIPCHECK(hipMalloc((void**)&Ad, sizeof(int)));
    HIPCHECK(hipMalloc((void**)&Bd, sizeof(int)));
    HIPCHECK(hipHostMalloc((void**)&A,sizeof(int)));

    // If the passed pointers do not match the kind, we should return a
    // hipErrorInvalidMemcpyDirection error
    HIPASSERT(hipMemcpy(Ad, A, sizeof(int), hipMemcpyDeviceToHost) == hipErrorInvalidMemcpyDirection);
    HIPASSERT(hipMemcpy(A,  Ad, sizeof(int), hipMemcpyHostToDevice) == hipErrorInvalidMemcpyDirection);
    HIPASSERT(hipMemcpy(Ad, Bd, sizeof(int), hipMemcpyHostToHost) == hipErrorInvalidMemcpyDirection);
#ifndef __HIP_PLATFORM_NVCC__
    HIPASSERT(hipMemcpy(A,  A, sizeof(int), hipMemcpyDeviceToDevice) == hipErrorInvalidMemcpyDirection);
#endif

    // nullptr passed as source or destination pointer
    HIPASSERT(hipSuccess != hipMemcpy(nullptr, A, sizeof(int), hipMemcpyHostToDevice));
    HIPASSERT(hipSuccess != hipMemcpy(Ad, nullptr, sizeof(int), hipMemcpyHostToDevice));

    HIPCHECK(hipFree(Ad));
    HIPCHECK(hipFree(Bd));
    HIPCHECK(hipFree(A));
    passed();
}
