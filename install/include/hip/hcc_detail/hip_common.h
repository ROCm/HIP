/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COMMON_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COMMON_H

#if defined(__HCC__)
#define __HCC_OR_HIP_CLANG__ 1
#define __HCC_ONLY__ 1
#define __HIP_CLANG_ONLY__ 0
#elif defined(__clang__) && defined(__HIP__)
#define __HCC_OR_HIP_CLANG__ 1
#define __HCC_ONLY__ 0
#define __HIP_CLANG_ONLY__ 1
#else
#define __HCC_OR_HIP_CLANG__ 0
#define __HCC_ONLY__ 0
#define __HIP_CLANG_ONLY__ 0
#endif

#endif // HIP_INCLUDE_HIP_HCC_DETAIL_HIP_COMMON_H
