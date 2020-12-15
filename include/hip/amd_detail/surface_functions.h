/*
Copyright (c) 2018 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H

#include <hip/amd_detail/hip_surface_types.h>

#define __SURFACE_FUNCTIONS_DECL__ static inline __device__
template <class T>
__SURFACE_FUNCTIONS_DECL__ void surf2Dread(T* data, hipSurfaceObject_t surfObj, int x, int y,
                                           int boundaryMode = hipBoundaryModeZero) {
    hipArray* arrayPtr = (hipArray*)surfObj;
    size_t width = arrayPtr->width;
    size_t height = arrayPtr->height;
    int32_t xOffset = x / sizeof(T);
    T* dataPtr = (T*)arrayPtr->data;
    if ((xOffset > width) || (xOffset < 0) || (y > height) || (y < 0)) {
        if (boundaryMode == hipBoundaryModeZero) {
            *data = 0;
        }
    } else {
        *data = *(dataPtr + y * width + xOffset);
    }
}

template <class T>
__SURFACE_FUNCTIONS_DECL__ void surf2Dwrite(T data, hipSurfaceObject_t surfObj, int x, int y,
                                            int boundaryMode = hipBoundaryModeZero) {
    hipArray* arrayPtr = (hipArray*)surfObj;
    size_t width = arrayPtr->width;
    size_t height = arrayPtr->height;
    int32_t xOffset = x / sizeof(T);
    T* dataPtr = (T*)arrayPtr->data;
    if (!((xOffset > width) || (xOffset < 0) || (y > height) || (y < 0))) {
        *(dataPtr + y * width + xOffset) = data;
    }
}

#endif
