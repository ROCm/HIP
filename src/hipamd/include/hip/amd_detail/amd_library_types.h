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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_LIBRARY_TYPES_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_LIBRARY_TYPES_H

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

#endif
