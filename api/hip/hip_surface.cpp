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

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"
#include <hip/hcc_detail/hip_surface_types.h>

namespace hip {

static amd::Monitor surfaceLock("Guards surface objects");

struct hipSurface {
  hipSurface(const hipResourceDesc* pResDesc): array(nullptr)
  {
    memcpy(&resDesc, pResDesc, sizeof(hipResourceDesc));
  }

  hipArray* array;
  hipResourceDesc resDesc;
};

static std::unordered_map<hipSurfaceObject_t, hipSurface*> surfaceHash;

};

using namespace hip;

hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                  const hipResourceDesc* pResDesc) {
  HIP_INIT_API(NONE, pSurfObject, pResDesc);

  hipSurface* pSurface = new hipSurface(pResDesc);
  assert(pSurface != nullptr);

  switch (pResDesc->resType) {
  case hipResourceTypeArray:
    pSurface->array = pResDesc->res.array.array;
    break;
  default:
    break;
  }
  hipSurfaceObject_t surfObj;
  hipError_t err = hipMalloc(reinterpret_cast<void**>(&surfObj), sizeof(hipArray));
  if (err != hipSuccess) {
    delete pSurface;
    HIP_RETURN(hipErrorOutOfMemory);
  }
  err = hipMemcpy(reinterpret_cast<void*>(surfObj), reinterpret_cast<void*>(pResDesc->res.array.array), sizeof(hipArray),
            hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    delete pSurface;
    hipFree(reinterpret_cast<void*>(surfObj));
    HIP_RETURN(err);
  }
  *pSurfObject = surfObj;

  amd::ScopedLock lock(surfaceLock);
  surfaceHash[*pSurfObject] = pSurface;

  HIP_RETURN(hipSuccess);
}


hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
  HIP_INIT_API(NONE, surfaceObject);

  amd::ScopedLock lock(surfaceLock);
  hipSurface* pSurface = surfaceHash[surfaceObject];
  if (pSurface != nullptr) {
    delete pSurface;
    surfaceHash.erase(surfaceObject);
    HIP_RETURN(hipFree(reinterpret_cast<void*>(surfaceObject)));
  }

  HIP_RETURN(hipErrorInvalidValue);
}
