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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_MEMORY_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_MEMORY_H

// Implementation of malloc and free device functions.
// HIP heap is implemented as a global array with fixed size. Users may define
// __HIP_SIZE_OF_PAGE and __HIP_NUM_PAGES to have a larger heap.

#if __HIP__ && __HIP_ENABLE_DEVICE_MALLOC__

// Size of page in bytes.
#ifndef __HIP_SIZE_OF_PAGE
#define __HIP_SIZE_OF_PAGE 64
#endif

// Total number of pages
#ifndef __HIP_NUM_PAGES
#define __HIP_NUM_PAGES (16 * 64 * 64)
#endif

#define __HIP_SIZE_OF_HEAP (__HIP_NUM_PAGES * __HIP_SIZE_OF_PAGE)

#if __HIP_DEVICE_COMPILE__
__attribute__((weak)) __device__ char __hip_device_heap[__HIP_SIZE_OF_HEAP];
__attribute__((weak)) __device__
    uint32_t __hip_device_page_flag[__HIP_NUM_PAGES];
#else
extern __device__ char __hip_device_heap[];
extern __device__ uint32_t __hip_device_page_flag[];
#endif

extern "C" inline __device__ void* __hip_malloc(size_t size) {
    char* heap = (char*)__hip_device_heap;
    if (size > __HIP_SIZE_OF_HEAP) {
        return (void*)nullptr;
    }
    uint32_t totalThreads =
        hipBlockDim_x * hipGridDim_x * hipBlockDim_y
        * hipGridDim_y * hipBlockDim_z * hipGridDim_z;
    uint32_t currentWorkItem = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x
        + (hipThreadIdx_y + hipBlockDim_y * hipBlockIdx_y) * hipBlockDim_x
        + (hipThreadIdx_z + hipBlockDim_z * hipBlockIdx_z) * hipBlockDim_x
        * hipBlockDim_y;

    uint32_t numHeapsPerWorkItem = __HIP_NUM_PAGES / totalThreads;
    uint32_t heapSizePerWorkItem = __HIP_SIZE_OF_HEAP / totalThreads;

    uint32_t stride = size / __HIP_SIZE_OF_PAGE;
    uint32_t start = numHeapsPerWorkItem * currentWorkItem;

    uint32_t k = 0;

    while (__hip_device_page_flag[k] > 0) {
        k++;
    }

    for (uint32_t i = 0; i < stride - 1; i++) {
        __hip_device_page_flag[i + start + k] = 1;
    }

    __hip_device_page_flag[start + stride - 1 + k] = 2;

    void* ptr = (void*)(heap
        + heapSizePerWorkItem * currentWorkItem + k * __HIP_SIZE_OF_PAGE);

    return ptr;
}

extern "C" inline __device__ void* __hip_free(void* ptr) {
    if (ptr == nullptr) {
        return nullptr;
    }

    uint32_t offsetByte = (uint64_t)ptr - (uint64_t)__hip_device_heap;
    uint32_t offsetPage = offsetByte / __HIP_SIZE_OF_PAGE;

    while (__hip_device_page_flag[offsetPage] != 0) {
        if (__hip_device_page_flag[offsetPage] == 2) {
            __hip_device_page_flag[offsetPage] = 0;
            offsetPage++;
            break;
        } else {
            __hip_device_page_flag[offsetPage] = 0;
            offsetPage++;
        }
    }

    return nullptr;
}

#endif

#endif // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_MEMORY_H
