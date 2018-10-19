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

#pragma once

// /*
// Half Math Functions
// */

#include "host_defines.h"

extern "C"
{
    __device__ __attribute__((const)) _Float16 __ocml_ceil_f16(_Float16);
    __device__ _Float16 __ocml_cos_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_floor_f16(_Float16);
    __device__ __attribute__((const))
    _Float16 __ocml_fma_f16(_Float16, _Float16, _Float16);
    __device__ __attribute__((const)) int __ocml_isinf_f16(_Float16);
    __device__ __attribute__((const)) int __ocml_isnan_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __llvm_amdgcn_rcp_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_rint_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
    __device__ _Float16 __ocml_sin_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_trunc_f16(_Float16);

    typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
    typedef short __2i16 __attribute__((ext_vector_type(2)));

    __device__ __attribute__((const)) __2f16 __ocml_ceil_2f16(__2f16);
    __device__ __2f16 __ocml_cos_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp10_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp2_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_floor_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_fma_2f16(__2f16, __2f16, __2f16);
    __device__ __attribute__((const)) __2i16 __ocml_isinf_2f16(__2f16);
    __device__ __attribute__((const)) __2i16 __ocml_isnan_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log10_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log2_2f16(__2f16);
    __device__ inline
    __2f16 __llvm_amdgcn_rcp_2f16(__2f16 x) // Not currently exposed by ROCDL.
    {
        return __2f16{__llvm_amdgcn_rcp_f16(x.x), __llvm_amdgcn_rcp_f16(x.y)};
    }
    __device__ __attribute__((const)) __2f16 __ocml_rint_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_rsqrt_2f16(__2f16);
    __device__ __2f16 __ocml_sin_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_sqrt_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_trunc_2f16(__2f16);
}