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

#include <map>

#include <string.h>

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

#include "hip_surface.h"

static std::map<hipSurfaceObject_t, hipSurface*> surfaceHash;

void saveSurfaceInfo(const hipSurface* pSurface, const hipResourceDesc* pResDesc) {
    if (pResDesc != nullptr) {
        memcpy((void*)&(pSurface->resDesc), (void*)pResDesc, sizeof(hipResourceDesc));
    }
}

// Surface Object APIs
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                  const hipResourceDesc* pResDesc) {
    HIP_INIT_API(hipCreateSurfaceObject, pSurfObject, pResDesc);
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hipSurface* pSurface = (hipSurface*)malloc(sizeof(hipSurface));
        if (pSurface != nullptr) {
            memset(pSurface, 0, sizeof(hipSurface));
            saveSurfaceInfo(pSurface, pResDesc);
        }

        switch (pResDesc->resType) {
            case hipResourceTypeArray:
                pSurface->array = pResDesc->res.array.array;
                break;
            default:
                break;
        }
        unsigned int* surfObj;
        hipMalloc((void**)&surfObj, sizeof(hipArray));
        hipMemcpy(surfObj, (void*)pResDesc->res.array.array, sizeof(hipArray),
                  hipMemcpyHostToDevice);
        *pSurfObject = (hipSurfaceObject_t)surfObj;
        surfaceHash[*pSurfObject] = pSurface;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
    HIP_INIT_API(hipDestroySurfaceObject, surfaceObject);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hipSurface* pSurface = surfaceHash[surfaceObject];
        if (pSurface != nullptr) {
            free(pSurface);
            surfaceHash.erase(surfaceObject);
        }
    }
    return ihipLogStatus(hip_status);
}
