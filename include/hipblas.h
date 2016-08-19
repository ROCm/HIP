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
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//!  This is the master include file for hipblas, wrapping around hcblas and cublas "version 1"
//

#pragma once

enum hipblasStatus_t {
  HIPBLAS_STATUS_SUCCESS,          // Function succeeds
  HIPBLAS_STATUS_NOT_INITIALIZED,  // HIPBLAS library not initialized
  HIPBLAS_STATUS_ALLOC_FAILED,     // resource allocation failed
  HIPBLAS_STATUS_INVALID_VALUE,    // unsupported numerical value was passed to function
  HIPBLAS_STATUS_MAPPING_ERROR,    // access to GPU memory space failed
  HIPBLAS_STATUS_EXECUTION_FAILED, // GPU program failed to execute
  HIPBLAS_STATUS_INTERNAL_ERROR,    // an internal HIPBLAS operation failed
  HIPBLAS_STATUS_NOT_SUPPORTED     // cublas supports this, but not hcblas
};
	
enum hipblasOperation_t {
	HIPBLAS_OP_N,
	HIPBLAS_OP_T,
	HIPBLAS_OP_C
};

// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:

#if defined(__HIP_PLATFORM_HCC__) and not defined (__HIP_PLATFORM_NVCC__)
#include <hcc_detail/hip_blas.h>
#elif defined(__HIP_PLATFORM_NVCC__) and not defined (__HIP_PLATFORM_HCC__)
#include <nvcc_detail/hip_blas.h>
#else 
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif 


	


