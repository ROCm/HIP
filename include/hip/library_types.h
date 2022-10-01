/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip/hip_common.h"

#if (defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) &&                            \
    !(defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__)) &&                       \
    !(defined(__HIP_PLATFORM_CLANG__) || defined(__HIP_PLATFORM_SPIRV__))
#define _USE_HIPCOMMON_LIBRARY_TYPES_

#elif (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__)) &&                      \
    !(defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) &&                           \
    !(defined(__HIP_PLATFORM_CLANG__) || defined(__HIP_PLATFORM_SPIRV__))
#include "library_types.h"

#elif (defined(__HIP_PLATFORM_CLANG__) || defined(__HIP_PLATFORM_SPIRV__)) &&                      \
    !(defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) &&                           \
    !(defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
#define _USE_HIPCOMMON_LIBRARY_TYPES_

#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__, __HIP_PLATFORM_NVIDIA__ or __HIP_PLATFORM_SPIRV__");
#endif // HIP PLATFORM SELECTION

#ifdef _USE_HIPCOMMON_LIBRARY_TYPES_

typedef enum hipDataType {
  HIP_R_16F = 2,
  HIP_R_32F = 0,
  HIP_R_64F = 1,
  HIP_C_16F = 6,
  HIP_C_32F = 4,
  HIP_C_64F = 5
} hipDataType;

typedef enum hipLibraryPropertyType {
  HIP_LIBRARY_MAJOR_VERSION,
  HIP_LIBRARY_MINOR_VERSION,
  HIP_LIBRARY_PATCH_LEVEL
} hipLibraryPropertyType;

#endif // _USE_HIPCOMMON_LIBRARY_TYPES_
