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

#ifndef HIPSTREAM_H
#define HIPSTREAM_H
#include "hip/hip_runtime.h"

#define NUM_STREAMS 4

/*
 * H2H - 1
 * H2D - 2
 * KER - 3
 * D2D - 4
 * D2H - 5
 */

template <typename T>
void H2HAsync(T* Dst, T* Src, size_t size, hipStream_t stream) {
    HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyHostToHost, stream));
}

template <typename T>
void H2DAsync(T* Dst, T* Src, size_t size, hipStream_t stream) {
    HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyHostToDevice, stream));
}

template <typename T>
void D2DAsync(T* Dst, T* Src, size_t size, hipStream_t stream) {
    HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyDeviceToDevice, stream));
}

template <typename T>
void D2HAsync(T* Dst, T* Src, size_t size, hipStream_t stream) {
    HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyDeviceToHost, stream));
}

template <typename T>
void H2H(T* Dst, T* Src, size_t size) {
    HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyHostToHost));
}

template <typename T>
void H2D(T* Dst, T* Src, size_t size) {
    HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyHostToDevice));
}

template <typename T>
void D2D(T* Dst, T* Src, size_t size) {
    HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyDeviceToDevice));
}

template <typename T>
void D2H(T* Dst, T* Src, size_t size) {
    HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyDeviceToHost));
}

template <typename T>
__global__ void Inc(T* In) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    In[tx] = In[tx] + 1;
}

template <typename T>
void initArrays(T** Ad, T** Ah, size_t N, bool usePinnedHost = false) {
    size_t NBytes = N * sizeof(T);
    if (Ad) {
        HIPCHECK(hipMalloc(Ad, NBytes));
    }
    if (usePinnedHost) {
        HIPCHECK(hipHostMalloc((void**)Ah, NBytes, hipHostMallocDefault));
    } else {
        *Ah = new T[N];
        HIPASSERT(*Ah != NULL);
    }
}

template <typename T>
void initArrays(T** Ad, size_t N, bool deviceMemory = false, bool usePinnedHost = false) {
    size_t NBytes = N * sizeof(T);
    if (deviceMemory) {
        HIPCHECK(hipMalloc(Ad, NBytes));
    } else {
        if (usePinnedHost) {
            HIPCHECK(hipHostMalloc((void**)Ad, NBytes, hipHostMallocDefault));
        } else {
            *Ad = new T[N];
            HIPASSERT(*Ad != NULL);
        }
    }
}

template <typename T>
void setArray(T* Array, int N, T val) {
    for (int i = 0; i < N; i++) {
        Array[i] = val;
    }
}


#endif
