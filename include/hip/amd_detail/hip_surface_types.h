/*
Copyright (c) 2015- present Advanced Micro Devices, Inc. All rights reserved.

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

/**
 *  @file  amd_detail/hip_surface_types.h
 *  @brief Defines surface types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_SURFACE_TYPES_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_SURFACE_TYPES_H

#include <hip/amd_detail/driver_types.h>

/**
 * An opaque value that represents a hip surface object
 */
typedef unsigned long long hipSurfaceObject_t;

/**
 * hip surface reference
 */
struct surfaceReference {
    hipSurfaceObject_t surfaceObject;
};

/**
 * hip surface boundary modes
 */
enum hipSurfaceBoundaryMode {
    hipBoundaryModeZero = 0,
    hipBoundaryModeTrap = 1,
    hipBoundaryModeClamp = 2
};

#endif /* !HIP_INCLUDE_HIP_AMD_DETAIL_HIP_SURFACE_TYPES_H */
