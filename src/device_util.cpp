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

__device__ long long int clock64() { return (long long int)hc::__cycle_u64(); };
__device__ clock_t clock() { return (clock_t)hc::__cycle_u64(); };

// abort
__device__ void abort() { return hc::abort(); }

// atomicAdd()
__device__ int atomicAdd(int* address, int val) { return hc::atomic_fetch_add(address, val); }
__device__ unsigned int atomicAdd(unsigned int* address, unsigned int val) {
    return hc::atomic_fetch_add(address, val);
}
__device__ unsigned long long int atomicAdd(unsigned long long int* address,
                                            unsigned long long int val) {
    return (long long int)hc::atomic_fetch_add((uint64_t*)address, (uint64_t)val);
}
__device__ float atomicAdd(float* address, float val) { return hc::atomic_fetch_add(address, val); }

// atomicSub()
__device__ int atomicSub(int* address, int val) { return hc::atomic_fetch_sub(address, val); }
__device__ unsigned int atomicSub(unsigned int* address, unsigned int val) {
    return hc::atomic_fetch_sub(address, val);
}

// atomicExch()
__device__ int atomicExch(int* address, int val) { return hc::atomic_exchange(address, val); }
__device__ unsigned int atomicExch(unsigned int* address, unsigned int val) {
    return hc::atomic_exchange(address, val);
}
__device__ unsigned long long int atomicExch(unsigned long long int* address,
                                             unsigned long long int val) {
    return (long long int)hc::atomic_exchange((uint64_t*)address, (uint64_t)val);
}
__device__ float atomicExch(float* address, float val) { return hc::atomic_exchange(address, val); }

// atomicMin()
__device__ int atomicMin(int* address, int val) { return hc::atomic_fetch_min(address, val); }
__device__ unsigned int atomicMin(unsigned int* address, unsigned int val) {
    return hc::atomic_fetch_min(address, val);
}
__device__ unsigned long long int atomicMin(unsigned long long int* address,
                                            unsigned long long int val) {
    return (long long int)hc::atomic_fetch_min((uint64_t*)address, (uint64_t)val);
}

// atomicMax()
__device__ int atomicMax(int* address, int val) { return hc::atomic_fetch_max(address, val); }
__device__ unsigned int atomicMax(unsigned int* address, unsigned int val) {
    return hc::atomic_fetch_max(address, val);
}
__device__ unsigned long long int atomicMax(unsigned long long int* address,
                                            unsigned long long int val) {
    return (long long int)hc::atomic_fetch_max((uint64_t*)address, (uint64_t)val);
}

// atomicCAS()
template <typename T>
__device__ T atomicCAS_impl(T* address, T compare, T val) {
    // the implementation assumes the atomic is lock-free and
    // has the same size as the non-atmoic equivalent type
    static_assert(sizeof(T) == sizeof(std::atomic<T>),
                  "size mismatch between atomic and non-atomic types");

    union {
        T* address;
        std::atomic<T>* atomic_address;
    } u;
    u.address = address;

    T expected = compare;

    // hcc should generate a system scope atomic CAS
    std::atomic_compare_exchange_weak_explicit(
        u.atomic_address, &expected, val, std::memory_order_acq_rel, std::memory_order_relaxed);
    return expected;
}

__device__ int atomicCAS(int* address, int compare, int val) {
    return atomicCAS_impl(address, compare, val);
}
__device__ unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val) {
    return atomicCAS_impl(address, compare, val);
}
__device__ unsigned long long int atomicCAS(unsigned long long int* address,
                                            unsigned long long int compare,
                                            unsigned long long int val) {
    return atomicCAS_impl(address, compare, val);
}

// atomicAnd()
__device__ int atomicAnd(int* address, int val) { return hc::atomic_fetch_and(address, val); }
__device__ unsigned int atomicAnd(unsigned int* address, unsigned int val) {
    return hc::atomic_fetch_and(address, val);
}
__device__ unsigned long long int atomicAnd(unsigned long long int* address,
                                            unsigned long long int val) {
    return (long long int)hc::atomic_fetch_and((uint64_t*)address, (uint64_t)val);
}

// atomicOr()
__device__ int atomicOr(int* address, int val) { return hc::atomic_fetch_or(address, val); }
__device__ unsigned int atomicOr(unsigned int* address, unsigned int val) {
    return hc::atomic_fetch_or(address, val);
}
__device__ unsigned long long int atomicOr(unsigned long long int* address,
                                           unsigned long long int val) {
    return (long long int)hc::atomic_fetch_or((uint64_t*)address, (uint64_t)val);
}

// atomicXor()
__device__ int atomicXor(int* address, int val) { return hc::atomic_fetch_xor(address, val); }
__device__ unsigned int atomicXor(unsigned int* address, unsigned int val) {
    return hc::atomic_fetch_xor(address, val);
}
__device__ unsigned long long int atomicXor(unsigned long long int* address,
                                            unsigned long long int val) {
    return (long long int)hc::atomic_fetch_xor((uint64_t*)address, (uint64_t)val);
}

// atomicInc
__device__ unsigned int atomicInc(unsigned int* address, unsigned int val) {
    return hc::__atomic_wrapinc(address, val);
}

// atomicDec
__device__ unsigned int atomicDec(unsigned int* address, unsigned int val) {
    return hc::__atomic_wrapdec(address, val);
}


// warp vote function __all __any __ballot
__device__ int __all(int input) { return hc::__all(input); }


__device__ int __any(int input) {
#ifdef NVCC_COMPAT
    if (hc::__any(input) != 0)
        return 1;
    else
        return 0;
#else
    return hc::__any(input);
#endif
}

__device__ unsigned long long int __ballot(int input) { return hc::__ballot(input); }

// warp shuffle functions
__device__ int __shfl(int input, int lane, int width) { return hc::__shfl(input, lane, width); }

__device__ int __shfl_up(int input, unsigned int lane_delta, int width) {
    return hc::__shfl_up(input, lane_delta, width);
}

__device__ int __shfl_down(int input, unsigned int lane_delta, int width) {
    return hc::__shfl_down(input, lane_delta, width);
}

__device__ int __shfl_xor(int input, int lane_mask, int width) {
    return hc::__shfl_xor(input, lane_mask, width);
}

__device__ float __shfl(float input, int lane, int width) { return hc::__shfl(input, lane, width); }

__device__ float __shfl_up(float input, unsigned int lane_delta, int width) {
    return hc::__shfl_up(input, lane_delta, width);
}

__device__ float __shfl_down(float input, unsigned int lane_delta, int width) {
    return hc::__shfl_down(input, lane_delta, width);
}

__device__ float __shfl_xor(float input, int lane_mask, int width) {
    return hc::__shfl_xor(input, lane_mask, width);
}

__host__ __device__ int min(int arg1, int arg2) {
    return (int)(hc::precise_math::fmin((float)arg1, (float)arg2));
}
__host__ __device__ int max(int arg1, int arg2) {
    return (int)(hc::precise_math::fmax((float)arg1, (float)arg2));
}

__device__ void* __get_dynamicgroupbaseptr() {
    return hc::get_dynamic_group_segment_base_pointer();
}

__host__ void* __get_dynamicgroupbaseptr() { return nullptr; }


__device__ void __threadfence_system(void) { std::atomic_thread_fence(std::memory_order_seq_cst); }