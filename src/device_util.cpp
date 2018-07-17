/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hc.hpp>
#include <grid_launch.h>
#include <hc_math.hpp>
#include "device_util.h"
#include "hip/hcc_detail/device_functions.h"
#include "hip/hip_runtime.h"
#include <atomic>

//=================================================================================================
/*
 Implementation of malloc and free device functions.

 This is the best place to put them because the device
 global variables need to be initialized at the start.
*/
__device__ char gpuHeap[SIZE_OF_HEAP];
__device__ uint32_t gpuFlags[NUM_PAGES];

__device__ void* __hip_hc_malloc(size_t size) {
    char* heap = (char*)gpuHeap;
    if (size > SIZE_OF_HEAP) {
        return (void*)nullptr;
    }
    uint32_t totalThreads =
        blockDim.x * gridDim.x * blockDim.y * gridDim.y * blockDim.z * gridDim.z;
    uint32_t currentWorkItem = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t numHeapsPerWorkItem = NUM_PAGES / totalThreads;
    uint32_t heapSizePerWorkItem = SIZE_OF_HEAP / totalThreads;

    uint32_t stride = size / SIZE_OF_PAGE;
    uint32_t start = numHeapsPerWorkItem * currentWorkItem;

    uint32_t k = 0;

    while (gpuFlags[k] > 0) {
        k++;
    }

    for (uint32_t i = 0; i < stride - 1; i++) {
        gpuFlags[i + start + k] = 1;
    }

    gpuFlags[start + stride - 1 + k] = 2;

    void* ptr = (void*)(heap + heapSizePerWorkItem * currentWorkItem + k * SIZE_OF_PAGE);

    return ptr;
}

__device__ void* __hip_hc_free(void* ptr) {
    if (ptr == nullptr) {
        return nullptr;
    }

    uint32_t offsetByte = (uint64_t)ptr - (uint64_t)gpuHeap;
    uint32_t offsetPage = offsetByte / SIZE_OF_PAGE;

    while (gpuFlags[offsetPage] != 0) {
        if (gpuFlags[offsetPage] == 2) {
            gpuFlags[offsetPage] = 0;
            offsetPage++;
            break;
        } else {
            gpuFlags[offsetPage] = 0;
            offsetPage++;
        }
    }

    return nullptr;
}


// loop unrolling
__device__ void* __hip_hc_memcpy(void* dst, const void* src, size_t size) {
    auto dstPtr = static_cast<uint8_t*>(dst);
    auto srcPtr = static_cast<const uint8_t*>(src);

    while (size >= 4u) {
        dstPtr[0] = srcPtr[0];
        dstPtr[1] = srcPtr[1];
        dstPtr[2] = srcPtr[2];
        dstPtr[3] = srcPtr[3];

        size -= 4u;
        srcPtr += 4u;
        dstPtr += 4u;
    }
    switch (size) {
        case 3:
            dstPtr[2] = srcPtr[2];
        case 2:
            dstPtr[1] = srcPtr[1];
        case 1:
            dstPtr[0] = srcPtr[0];
    }

    return dst;
}

__device__ void* __hip_hc_memset(void* dst, uint8_t val, size_t size) {
    auto dstPtr = static_cast<uint8_t*>(dst);

    while (size >= 4u) {
        dstPtr[0] = val;
        dstPtr[1] = val;
        dstPtr[2] = val;
        dstPtr[3] = val;

        size -= 4u;
        dstPtr += 4u;
    }
    switch (size) {
        case 3:
            dstPtr[2] = val;
        case 2:
            dstPtr[1] = val;
        case 1:
            dstPtr[0] = val;
    }

    return dst;
}

// abort
__device__ void abort() { return hc::abort(); }

__host__ void* __get_dynamicgroupbaseptr() { return nullptr; }
