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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM hcc
 * RUN: %t EXCLUDE_HIP_PLATFORM nvcc
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

template <typename T>
__global__ void testExternSharedKernel(const T* A_d, const T* B_d, T* C_d,
                                       size_t numElements, size_t groupElements) {
    // declare dynamic shared memory
#if defined(__HIP_PLATFORM_HCC__)
    HIP_DYNAMIC_SHARED(T, sdata)
#else
    HIP_DYNAMIC_SHARED(__align__(sizeof(T)) unsigned char, my_sdata)
    T* sdata = reinterpret_cast<T*>(my_sdata);
#endif

    size_t gid = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t tid = threadIdx.x;

    // initialize dynamic shared memory
    if (tid < groupElements) {
        sdata[tid] = static_cast<T>(tid);
    }

    // prefix sum inside dynamic shared memory
    if (groupElements >= 512) {
        if (tid >= 256) {
            sdata[tid] += sdata[tid - 256];
        }
        __syncthreads();
    }
    if (groupElements >= 256) {
        if (tid >= 128) {
            sdata[tid] += sdata[tid - 128];
        }
        __syncthreads();
    }
    if (groupElements >= 128) {
        if (tid >= 64) {
            sdata[tid] += sdata[tid - 64];
        }
        __syncthreads();
    }
    if (groupElements >= 64) {
        sdata[tid] += sdata[tid - 32];
    }
    __syncthreads();
    if (groupElements >= 32) {
        sdata[tid] += sdata[tid - 16];
    }
    __syncthreads();
    if (groupElements >= 16) {
        sdata[tid] += sdata[tid - 8];
    }
    __syncthreads();
    if (groupElements >= 8) {
        sdata[tid] += sdata[tid - 4];
    }
    __syncthreads();
    if (groupElements >= 4) {
        sdata[tid] += sdata[tid - 2];
    }
    __syncthreads();
    if (groupElements >= 2) {
        sdata[tid] += sdata[tid - 1];
    }
    __syncthreads();

    C_d[gid] = A_d[gid] + B_d[gid] + sdata[tid % groupElements];
}

template <typename T>
void testExternShared(size_t N, size_t groupElements) {
    size_t Nbytes = N * sizeof(T);

    T *A_d, *B_d, *C_d;
    T *A_h, *B_h, *C_h;

    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

    // printf("blocksPerCU: %d\nthreadsPerBlock: %d\nN: %zu\n", blocksPerCU, threadsPerBlock, N);

    HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    // calculate the amount of dynamic shared memory required
    size_t groupMemBytes = groupElements * sizeof(T);

    // launch kernel with dynamic shared memory
    hipLaunchKernelGGL(HIP_KERNEL_NAME(testExternSharedKernel<T>), dim3(blocks), dim3(threadsPerBlock),
                    groupMemBytes, 0, A_d, B_d, C_d, N, groupElements);

    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    // verify
    for (size_t i = 0; i < N; ++i) {
        size_t tid = (i % groupElements);
        T sumFromSharedMemory = static_cast<T>(tid * (tid + 1) / 2);
        T expected = A_h[i] + B_h[i] + sumFromSharedMemory;
        if (C_h[i] != expected) {
            std::cout << std::fixed << std::setprecision(32);
            std::cout << "At " << i << std::endl;
            std::cout << "  Computed:" << C_h[i] << std::endl;
            std::cout << "  Expected:" << expected << std::endl;
            std::cout << sumFromSharedMemory << std::endl;
            std::cout << A_h[i] << std::endl;
            std::cout << B_h[i] << std::endl;

            failed("Failed at index:%zu\n", i);
        }
    }

    HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);

    // printf("info: set device to %d\n", p_gpuDevice);
    HIPCHECK(hipSetDevice(p_gpuDevice));

    testExternShared<float>(1024, 4);
    testExternShared<float>(1024, 8);
    testExternShared<float>(1024, 16);
    testExternShared<float>(1024, 32);
    testExternShared<float>(1024, 64);

    testExternShared<float>(65536, 4);
    testExternShared<float>(65536, 8);
    testExternShared<float>(65536, 16);
    testExternShared<float>(65536, 32);
    testExternShared<float>(65536, 64);

    testExternShared<double>(1024, 4);
    testExternShared<double>(1024, 8);
    testExternShared<double>(1024, 16);
    testExternShared<double>(1024, 32);
    testExternShared<double>(1024, 64);

    testExternShared<double>(65536, 4);
    testExternShared<double>(65536, 8);
    testExternShared<double>(65536, 16);
    testExternShared<double>(65536, 32);
    testExternShared<double>(65536, 64);

    passed();
}
