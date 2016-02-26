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

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified 
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well:
//
//! Both paths support rich C++ features including classes, templates, lambdas, etc.
//! Runtime API is C
//! Memory management is based on pure pointers and resembles malloc/free/copy.
//
//! hip_runtime.h     : includes everything in hip_api.h, plus math builtins and kernel launch macros.
//! hip_runtime_api.h : Defines HIP API.  This is a C header file and does not use any C++ features.

#pragma once

// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef __cplusplus
#include <thread>
#endif


#include <hip_common.h>

#if defined(__HIP_PLATFORM_HCC__) and not defined (__HIP_PLATFORM_NVCC__)
#include <hcc_detail/hip_runtime.h>
#elif defined(__HIP_PLATFORM_NVCC__) and not defined (__HIP_PLATFORM_HCC__)
#include <nvcc_detail/hip_runtime.h>
#else 
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif 


#include <hip_runtime_api.h>
#include <hip_vector_types.h>

