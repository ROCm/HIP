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

#ifndef HIP_HCC_DETAIL_DEVICE_FUNCTIONS_H
#define HIP_HCC_DETAIL_DEVICE_FUNCTIONS_H

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

__device__ float __int_as_float (int x);

__device__ double __hiloint2double (int hi, int lo);

__device__ char4 __hip_hc_add8pk(char4, char4);
__device__ char4 __hip_hc_sub8pk(char4, char4);
__device__ char4 __hip_hc_mul8pk(char4, char4);

extern __device__ double  __longlong_as_double(long long int x);
extern __device__ long long int __double_as_longlong(double x);

#endif
