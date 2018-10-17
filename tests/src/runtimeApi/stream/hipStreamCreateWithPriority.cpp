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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#define MEMCPYSIZE 64*1024*1024
#define NUMITERS   2
#define GRIDSIZE   1024
#define BLOCKSIZE  256

// helper rountine to initialize memory
template <typename T>
void mem_init(T* buf, size_t n)
{
    for (int i = 0; i < n; i++)
    {
        buf[i] = i;
    }
}

// kernel to copy n elements from src to dst
template <typename T>
__global__ void memcpy_kernel(T* dst, T* src, size_t n)
{
    int num = gridDim.x * blockDim.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = id; i < n; i += num)
    {
        dst[i] = src[i];
    }
}

template <typename T>
void runTest()
{
    size_t size = NUMITERS*MEMCPYSIZE;

    // get the range of priorities available
    #define OP(x) \
        int priority_##x; \
        bool enable_priority_##x = false;
    OP(low)
    OP(normal)
    OP(high)
    #undef OP
    HIPCHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    printf("HIP stream priority range - low: %d to high: %d\n", priority_low, priority_high);

    // Check if priorities are indeed supported
    if ((priority_low - priority_high) == 0) { passed(); }
    
    // Enable/disable priorities based on number of available priority levels
    enable_priority_low = true;
    enable_priority_high = true;
    if ((priority_low - priority_high) > 1) enable_priority_normal = true;
    if (enable_priority_normal) priority_normal = ((priority_low - priority_high) / 2);

    // create streams with highest and lowest available priorities
    #define OP(x) \
        hipStream_t stream_##x; \
        if (enable_priority_##x) { \
            HIPCHECK(hipStreamCreateWithPriority(&stream_##x, hipStreamDefault, priority_##x)); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // allocate and initialise host source and destination buffers
    #define OP(x) \
        T* src_h_##x; \
        T* dst_h_##x; \
        if (enable_priority_##x) { \
            src_h_##x = (T*)malloc(size); \
            if (src_h_##x == NULL) { printf("src_h_%s malloc failed!\n", #x); exit(-1); } \
            mem_init<T>(src_h_##x, (size / sizeof(T))); \
            dst_h_##x = (T*)malloc(size); \
            if (dst_h_##x == NULL) { printf("dst_h_%s malloc failed!\n", #x); exit(-1); } \
            memset(dst_h_##x, 0, size); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // allocate and initialize device source and destination buffers
    #define OP(x) \
        T* src_d_##x; \
        T* dst_d_##x; \
        if (enable_priority_##x) { \
            HIPCHECK(hipMalloc(&src_d_##x, size)); \
            HIPCHECK(hipMemcpy(src_d_##x, src_h_##x, size, hipMemcpyHostToDevice)); \
            HIPCHECK(hipMalloc(&dst_d_##x, size)); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // create events for measuring time spent in kernel execution
    #define OP(x) \
        hipEvent_t event_start_##x; \
        hipEvent_t event_end_##x; \
        if (enable_priority_##x) { \
            HIPCHECK(hipEventCreate(&event_start_##x)); \
            HIPCHECK(hipEventCreate(&event_end_##x)); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // record start events for each of the priority streams
    #define OP(x) \
        if (enable_priority_##x) { \
            HIPCHECK(hipEventRecord(event_start_##x, stream_##x)); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // launch kernels repeatedly on each of the prioritiy streams
    for (int i = 0; i < size; i += MEMCPYSIZE)
    {
        int j = i / sizeof(T);
        #define OP(x) \
            if (enable_priority_##x) { \
                hipLaunchKernelGGL((memcpy_kernel<T>), dim3(GRIDSIZE), dim3(BLOCKSIZE), 0, stream_##x, dst_d_##x + j, src_d_##x + j, (MEMCPYSIZE / sizeof(T))); \
            }
        OP(low)
        OP(normal)
        OP(high)
        #undef OP
    }

    // record end events for each of the priority streams
    #define OP(x) \
        if (enable_priority_##x) { \
            HIPCHECK(hipEventRecord(event_end_##x, stream_##x)); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // synchronize events for each of the priority streams
    #define OP(x) \
        if (enable_priority_##x) { \
            HIPCHECK(hipEventSynchronize(event_end_##x)); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // compute time spent for memcpy in each stream
    #define OP(x) \
        float time_spent_##x; \
        if (enable_priority_##x) { \
            HIPCHECK(hipEventElapsedTime(&time_spent_##x, event_start_##x, event_end_##x)); \
            printf("time spent for memcpy in %6s priority stream: %.3lf ms\n", #x, time_spent_##x); \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // sanity check
    #define OP(x) \
        if (enable_priority_##x) { \
            HIPCHECK(hipMemcpy(dst_h_##x, dst_d_##x, size, hipMemcpyDeviceToHost)); \
            if (memcmp(dst_h_##x, src_h_##x, size) != 0) { printf("memcmp for %s failed!\n", #x); exit(-1); } \
        }
    OP(low)
    OP(normal)
    OP(high)
    #undef OP

    // validate that stream priorities are working as expected
    #define OP(x, y) \
        if (enable_priority_##x && enable_priority_##y) { \
            if (time_spent_##x < time_spent_##y) { printf("FAILED!"); exit(-1); } \
        }
    OP(low, normal)
    OP(normal, high)
    OP(low, high)
    #undef OP
    passed();
}

int main(int argc, char **argv)
{
    HipTest::parseStandardArguments(argc, argv, false);
    runTest<int>();
}
