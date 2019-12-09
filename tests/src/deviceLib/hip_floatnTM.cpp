/*
Copyright (c) 2015-2019 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <type_traits>
#include <random>
#include "test_common.h"


template <typename T, typename M>
int count() {
    return sizeof(T) / sizeof(M);
}

template <typename N>
void cpuJitter2(N& b) {
    b.x++;
    b.y++;
    b.x += b.y;
}

template <typename M, typename N>
void cpuJitter3(M& b) {
    cpuJitter2<N>(*reinterpret_cast<N*>(&b));
    b.z++;
    b.x = b.y + b.z;
}

template <typename T, typename M, typename N>
void cpuJitter4(T& b) {
    cpuJitter3<M, N>(*reinterpret_cast<M*>(&b));
    b.w++;
    b.x = b.w + b.y + b.z;
}

// Rotate x,y,z,w by 1
template <typename N>
void cpuRotate2(N& a, N& b) {
    b.x = a.y;
    b.y = a.x;
    cpuJitter2<N>(b);
}

template <typename M, typename N>
void cpuRotate3(M& a, M& b) {
    cpuRotate2<N>(*reinterpret_cast<N*>(&a), *reinterpret_cast<N*>(&b));
    b.y = a.z;
    b.z = a.x;
    cpuJitter3<N>(b);
}

template <typename T, typename M, typename N>
void cpuRotate4(T& a, T& b) {
    cpuRotate3<M, N>(*reinterpret_cast<M*>(&a), *reinterpret_cast<M*>(&b));
    b.y = a.z;
    b.z = a.w;
    b.w = a.x;
    cpuJitter4<M, N>(b);
}

template <typename T>
void cpuRotate2(T* a, T* b, int size) {
    for (int i = 0; i < size; i++) {
        cpuRotate2<T>(a[i], b[i]);
    }
}

template <typename T, typename M>
void cpuRotate3(T* a, T* b, int size) {
    for (int i = 0; i < size; i++) {
        cpuRotate<T, M>(a[i], b[i]);
    }
}

template <typename T, typename M, typename N>
void cpuRotate4(T* a, T* b, int size) {
    for (int i = 0; i < size; i++) {
        cpuRotate4<T, M, N>(a[i], b[i]);
    }
}

inline int getRandomNumber(int min = INT_MIN, int max = INT_MAX) {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> gen(min, max);
    return gen(rng);
}

inline float getRandomFloat() { return getRandomNumber() / getRandomNumber(); }

template <typename T>
void fillMatrix2(T* a, int size) {
    for (int i = 0; i < size; i++) {
        T t;
        t.x = getRandomFloat();
        t.y = getRandomFloat();
        a[i] = t;
    }
}

template <typename T>
void fillMatrix3(T* a, int size) {
    for (int i = 0; i < size; i++) {
        T t;
        t.x = getRandomFloat();
        t.y = getRandomFloat();
        t.z = getRandomFloat();
        a[i] = t;
    }
}

template <typename T>
void fillMatrix4(T* a, int size) {
    for (int i = 0; i < size; i++) {
        T t;
        t.x = getRandomFloat();
        t.y = getRandomFloat();
        t.z = getRandomFloat();
        t.w = getRandomFloat();
        a[i] = t;
    }
}

// a[i][j] += (a[i][j] * b[j][i]);
template <typename T>
void matAcc(T* a, T* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += (a[i] * b[i]);
    }
}

template <typename T>
void dcopy(T* a, T* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = b[i];
    }
}

template <typename T>
bool isEqual(T* a, T* b, int size) {
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
__global__ void gMatAcc(T* a, T* b, int size) {
    int i = threadIdx.x;

    if (i >= size) return;
    a[i] += (a[i] * b[i]);
}

template <typename T, typename S, typename A, typename B, typename C>
bool testType(int msize) {
    T *fa, *fb, *fc;
    fa = new T[msize];
    fb = new T[msize];
    fc = new T[msize];

    T *d_fa, *d_fb;

    int c = count<T, S>();
    if (c == 4) {
        fillMatrix4<T>(fa, msize);
        cpuRotate4<A, B, C>(fa, fb, msize);
    } else if (c == 3) {
        fillMatrix3<T>(fa, msize);
        cpuRotate3<B, C>(fa, fb, msize);
    } else if (c == 2) {
        fillMatrix2<T>(fa, msize);
        cpuRotate2<C>(fa, fb, msize);
    } else {
        failed("Invalid Size\n");
    }

    hipMalloc(&d_fa, sizeof(T) * msize);
    hipMalloc(&d_fb, sizeof(T) * msize);

    hipMemcpy(d_fa, fa, sizeof(T) * msize, hipMemcpyHostToDevice);
    hipMemcpy(d_fb, fb, sizeof(T) * msize, hipMemcpyHostToDevice);

    matAcc<T>(fa, fb, msize);

    hipLaunchKernelGGL(gMatAcc, 1, msize, 0, 0, d_fa, d_fb, msize);

    hipMemcpy(fc, d_fa, sizeof(T) * msize, hipMemcpyDeviceToHost);

    delete[] fa;
    delete[] fb;
    delete[] fc;
    hipFree(d_fa);
    hipFree(d_fb);

    if (!isEqual<T>(fa, fc, msize)) {
        failed("Failed for:: ");
        failed(typeid(T).name().c_str());
    }
    return true;
}

int main() {
    const int msize = 500;
    testType<float2, float, float4, float3, float2>();
    passed();
}
