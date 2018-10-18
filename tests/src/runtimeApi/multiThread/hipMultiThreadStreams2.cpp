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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN: %t
 * HIT_END
 */

#include <iostream>
#include "test_common.h"
#include <thread>
#define N 1000


template <typename T>
__global__ void Inc(T* Array) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    Array[tx] = Array[tx] + T(1);
}

void run1(size_t size, hipStream_t stream) {
    float *Ah, *Bh, *Cd, *Dd, *Eh;

    HIPCHECK(hipHostMalloc((void**)&Ah, size, hipHostMallocDefault));
    HIPCHECK(hipHostMalloc((void**)&Bh, size, hipHostMallocDefault));
    HIPCHECK(hipMalloc(&Cd, size));
    HIPCHECK(hipMalloc(&Dd, size));
    HIPCHECK(hipHostMalloc((void**)&Eh, size, hipHostMallocDefault));

    for (int i = 0; i < N; i++) {
        Ah[i] = 1.0f;
    }

    HIPCHECK(hipMemcpyAsync(Bh, Ah, size, hipMemcpyHostToHost, stream));
    HIPCHECK(hipMemcpyAsync(Cd, Bh, size, hipMemcpyHostToDevice, stream));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Inc), dim3(N / 500), dim3(500), 0, stream, Cd);
    HIPCHECK(hipMemcpyAsync(Dd, Cd, size, hipMemcpyDeviceToDevice, stream));
    HIPCHECK(hipMemcpyAsync(Eh, Dd, size, hipMemcpyDeviceToHost, stream));
    HIPCHECK(hipDeviceSynchronize());
    HIPASSERT(Eh[10] == Ah[10] + 1.0f);
}


void run(size_t size, hipStream_t stream1, hipStream_t stream2) {
    float *Ah, *Bh, *Cd, *Dd, *Eh;
    float *Ahh, *Bhh, *Cdd, *Ddd, *Ehh;

    HIPCHECK(hipHostMalloc((void**)&Ah, size, hipHostMallocDefault));
    HIPCHECK(hipHostMalloc((void**)&Bh, size, hipHostMallocDefault));
    HIPCHECK(hipMalloc(&Cd, size));
    HIPCHECK(hipMalloc(&Dd, size));
    HIPCHECK(hipHostMalloc((void**)&Eh, size, hipHostMallocDefault));
    HIPCHECK(hipHostMalloc((void**)&Ahh, size, hipHostMallocDefault));
    HIPCHECK(hipHostMalloc((void**)&Bhh, size, hipHostMallocDefault));
    HIPCHECK(hipMalloc(&Cdd, size));
    HIPCHECK(hipMalloc(&Ddd, size));
    HIPCHECK(hipHostMalloc((void**)&Ehh, size, hipHostMallocDefault));

    HIPCHECK(hipMemcpyAsync(Bh, Ah, size, hipMemcpyHostToHost, stream1));
    HIPCHECK(hipMemcpyAsync(Bhh, Ahh, size, hipMemcpyHostToHost, stream2));
    HIPCHECK(hipMemcpyAsync(Cd, Bh, size, hipMemcpyHostToDevice, stream1));
    HIPCHECK(hipMemcpyAsync(Cdd, Bhh, size, hipMemcpyHostToDevice, stream2));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Inc), dim3(N / 500), dim3(500), 0, stream1, Cd);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Inc), dim3(N / 500), dim3(500), 0, stream2, Cdd);
    HIPCHECK(hipMemcpyAsync(Dd, Cd, size, hipMemcpyDeviceToDevice, stream1));
    HIPCHECK(hipMemcpyAsync(Ddd, Cdd, size, hipMemcpyDeviceToDevice, stream2));
    HIPCHECK(hipMemcpyAsync(Eh, Dd, size, hipMemcpyDeviceToHost, stream1));
    HIPCHECK(hipMemcpyAsync(Ehh, Ddd, size, hipMemcpyDeviceToHost, stream2));
    HIPCHECK(hipDeviceSynchronize());
    HIPASSERT(Eh[10] = Ah[10] + 1.0f);
    HIPASSERT(Ehh[10] = Ahh[10] + 1.0f);

    HIPCHECK(hipHostFree(Ah));
    HIPCHECK(hipHostFree(Bh));
    HIPCHECK(hipHostFree(Eh));
    HIPCHECK(hipHostFree(Ahh));
    HIPCHECK(hipHostFree(Bhh));
    HIPCHECK(hipHostFree(Ehh));
    HIPCHECK(hipFree(Cd));
    HIPCHECK(hipFree(Dd));
    HIPCHECK(hipFree(Cdd));
    HIPCHECK(hipFree(Ddd));
}

int main(int argc, char** argv) {
    iterations = 100;

    HipTest::parseStandardArguments(argc, argv, true);


    hipStream_t stream[3];
    for (int i = 0; i < 3; i++) {
        HIPCHECK(hipStreamCreate(&stream[i]));
    }

    const size_t size = N * sizeof(float);

    for (int i = 0; i < iterations; i++) {
        std::thread t1(run1, size, stream[0]);
        std::thread t2(run1, size, stream[0]);
        std::thread t3(run, size, stream[1], stream[2]);
        t1.join();
        //	std::cout<<"T1"<<std::endl;
        t2.join();
        //	std::cout<<"T2"<<std::endl;
        t3.join();
    }
    passed();
}
