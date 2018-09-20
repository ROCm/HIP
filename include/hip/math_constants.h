/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_MATH_CONSTANTS_H
#define HIP_INCLUDE_HIP_MATH_CONSTANTS_H


// Create architecture (hcc / nvcc) specific headers if/when a constant value
// needs to be defined differently for either of the two

/* single precision constants */

#define HIPRT_INF_F        __int_as_float(0x7f800000)
#define HIPRT_NAN_F        __int_as_float(0x7fffffff)
#define HIPRT_MIN_DENORM_F __int_as_float(0x00000001)
#define HIPRT_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define HIPRT_NEG_ZERO_F   __int_as_float(0x80000000)
#define HIPRT_ZERO_F       0.0f
#define HIPRT_ONE_F        1.0f

/* double precision constants */
#define HIPRT_INF          __hiloint2double(0x7ff00000, 0x00000000)
#define HIPRT_NAN          __hiloint2double(0xfff80000, 0x00000000)


#endif // HIP_INCLUDE_HIP_MATH_CONSTANTS_H
