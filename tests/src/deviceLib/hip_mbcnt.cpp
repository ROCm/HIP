/*
Copyright (c) 2015-2018 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

__global__ void HIP_kernel(unsigned int* mbcnt_lo, unsigned int* mbcnt_hi, unsigned int* lane_id) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    mbcnt_lo[x] = __mbcnt_lo(0xFFFFFFFF, 0);
    mbcnt_hi[x] = __mbcnt_hi(0xFFFFFFFF, 0);
    lane_id[x] = __lane_id();
}
 
using namespace std;

int main() {

    unsigned int* device_mbcnt_lo;
    unsigned int* device_mbcnt_hi;
    unsigned int* device_lane_id;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    cout << "hip Device prop succeeded " << endl;

    constexpr unsigned int wave_size = 64;
    constexpr unsigned int num_waves_per_block = 2;
    constexpr unsigned int num_threads_per_block = wave_size * num_waves_per_block;
    constexpr unsigned int num_blocks = 2;
    constexpr unsigned int num_threads = num_threads_per_block * num_blocks;
    constexpr size_t buffer_size = num_threads * sizeof(unsigned int);

    HIP_ASSERT(hipMalloc((void**)&device_mbcnt_lo, buffer_size));
    HIP_ASSERT(hipMalloc((void**)&device_mbcnt_hi, buffer_size));
    HIP_ASSERT(hipMalloc((void**)&device_lane_id, buffer_size));

    hipLaunchKernelGGL(HIP_kernel, dim3(num_blocks),
                    dim3(num_threads_per_block), 0, 0, device_mbcnt_lo, device_mbcnt_hi, device_lane_id);

    unsigned int* host_mbcnt_lo = (unsigned int*) malloc(buffer_size);
    unsigned int* host_mbcnt_hi = (unsigned int*) malloc(buffer_size);
    unsigned int* host_lane_id = (unsigned int*) malloc(buffer_size);

    HIP_ASSERT(hipMemcpy(host_mbcnt_lo, device_mbcnt_lo, buffer_size, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(host_mbcnt_hi, device_mbcnt_hi, buffer_size, hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(host_lane_id, device_lane_id, buffer_size, hipMemcpyDeviceToHost));

    // verify the results
    int mbcnt_lo_errors = 0;
    int mbcnt_hi_errors = 0;
    int lane_id_errors = 0;
    for (unsigned int i = 0; i < num_threads; i++) {
        unsigned int this_lane_id = i % wave_size;
        unsigned int this_mbcnt_lo = this_lane_id >= 32 ? 32 : this_lane_id;
        unsigned int this_mbcnt_hi = this_lane_id < 32 ? 0 : (this_lane_id - 32);

        if (host_mbcnt_lo[i] != this_mbcnt_lo)
            mbcnt_lo_errors++;

        if (host_mbcnt_hi[i] != this_mbcnt_hi)
            mbcnt_hi_errors++;

        if (host_lane_id[i] != this_lane_id)
            lane_id_errors++;
    }

    if (mbcnt_lo_errors == 0)
      cout << "__mbcnt_lo() PASSED!" << endl;
    else
      cout << "__mbcnt_lo() FAILED!" << endl;


    if (mbcnt_hi_errors == 0)
      cout << "__mbcnt_hi() PASSED!" << endl;
    else
      cout << "__mbcnt_hi() FAILED!" << endl;

    if (lane_id_errors == 0)
      cout << "__lane_id() PASSED!" << endl;
    else
      cout << "__lane_id() FAILED!" << endl;

    HIP_ASSERT(hipFree(device_mbcnt_lo));
    HIP_ASSERT(hipFree(device_mbcnt_hi));
    HIP_ASSERT(hipFree(device_lane_id));

    free(host_mbcnt_lo);
    free(host_mbcnt_hi);
    free(host_lane_id);

    return mbcnt_lo_errors + mbcnt_hi_errors + lane_id_errors;
}
