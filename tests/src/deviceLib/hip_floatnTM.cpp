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
int count(T& a) {
    return sizeof(a) / sizeof(T);
}
/*
template <typename T>
void cpuJitter(T& b) {}

template <>
void cpuJitter<float2>(float2& b) {
   b.x++;
   b.y++;
   b.x += b.y;
}

template <>
void cpuJitter<float3>(float3& b) {
   cpuJitter<float2>(*reinterpret_cast<float2*>(&b));
   b.z++;
   b.x = b.y + b.z;
}

template <>
void cpuJitter<float4>(float4& b) {
   cpuJitter<float2>(*reinterpret_cast<float2*>(&b));
   b.w++;
   b.x = b.w + b.y + b.z;
}
*/

template <typename T, size_t N = 2>
void cpuJitter(T& b) {
    b.x++;
    b.y++;
    b.x += b.y;
}

template <typename T, size_t N = 3>
void cpuJitter(T& b) {
    cpuJitter<T, N - 1>(*reinterpret_cast<T*>(&b));
    b.z++;
    b.x = b.y + b.z;
}

template <typename T, size_t N = 4>
void cpuJitter(T& b) {
    cpuJitter<T, N - 1>(*reinterpret_cast<T*>(&b));
    b.w++;
    b.x = b.w + b.y + b.z;
}

// Rotate x,y,z,w by 1
template <typename T>
void cpuRotate(T& a, T& b) {}

template <>
void cpuRotate<float2>(float2& a, float2& b) {
    b.x = a.y;
    b.y = a.x;
    cpuJitter<float2, 2>(b);
}

template <>
void cpuRotate<float3>(float3& a, float3& b) {
    cpuRotate<float2>(*reinterpret_cast<float2*>(&a), *reinterpret_cast<float2*>(&b));
    b.y = a.z;
    b.z = a.x;
    cpuJitter<float3, 3>(b);
}

template <>
void cpuRotate<float4>(float4& a, float4& b) {
    cpuRotate<float2>(*reinterpret_cast<float2*>(&a), *reinterpret_cast<float2*>(&b));
    b.y = a.z;
    b.z = a.w;
    b.w = a.x;
    cpuJitter<float4, 4>(b);
}

template <typename T>
void cpuRotate(T* a, T* b, int size) {
    for (int i = 0; i < size; i++) {
        cpuRotate(a[i], b[i]);
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
void fillMatrix(T* a, int size) {}

template <>
void fillMatrix<float2>(float2* a, int size) {
    for (int i = 0; i < size; i++) {
        float2 t;
        t.x = getRandomFloat();
        t.y = getRandomFloat();
        a[i] = t;
    }
}

template <>
void fillMatrix<float3>(float3* a, int size) {
    for (int i = 0; i < size; i++) {
        float3 t;
        t.x = getRandomFloat();
        t.y = getRandomFloat();
        t.z = getRandomFloat();
        a[i] = t;
    }
}

template <>
void fillMatrix<float4>(float4* a, int size) {
    for (int i = 0; i < size; i++) {
        float4 t;
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

int main() {
    const int msize = 500;
    {
        float2 *fa, *fb, *fc;
        fa = new float2[msize];
        fb = new float2[msize];
        fc = new float2[msize];

        float2 *d_fa, *d_fb;

        fillMatrix<float2>(fa, msize);

        cpuRotate<float2>(fa, fb, msize);

        hipMalloc(&d_fa, sizeof(float2) * msize);
        hipMalloc(&d_fb, sizeof(float2) * msize);

        hipMemcpy(d_fa, fa, sizeof(float2) * msize, hipMemcpyHostToDevice);
        hipMemcpy(d_fb, fb, sizeof(float2) * msize, hipMemcpyHostToDevice);

        matAcc<float2>(fa, fb, msize);

        hipLaunchKernelGGL(gMatAcc, 1, msize, 0, 0, d_fa, d_fb, msize);

        hipMemcpy(fc, d_fa, sizeof(float2) * msize, hipMemcpyDeviceToHost);

        if (!isEqual<float2>(fa, fc, msize)) {
            failed("Fail float2");
        }

        delete[] fa;
        delete[] fb;
        delete[] fc;
        hipFree(d_fa);
        hipFree(d_fb);
    }

    {
        float3 *fa, *fb, *fc;
        fa = new float3[msize];
        fb = new float3[msize];
        fc = new float3[msize];

        float3 *d_fa, *d_fb;

        fillMatrix<float3>(fa, msize);

        cpuRotate<float3>(fa, fb, msize);

        hipMalloc(&d_fa, sizeof(float3) * msize);
        hipMalloc(&d_fb, sizeof(float3) * msize);

        hipMemcpy(d_fa, fa, sizeof(float3) * msize, hipMemcpyHostToDevice);
        hipMemcpy(d_fb, fb, sizeof(float3) * msize, hipMemcpyHostToDevice);

        matAcc<float3>(fa, fb, msize);

        hipLaunchKernelGGL(gMatAcc, 1, msize, 0, 0, d_fa, d_fb, msize);

        hipMemcpy(fc, d_fa, sizeof(float3) * msize, hipMemcpyDeviceToHost);

        if (!isEqual<float3>(fa, fc, msize)) {
            failed("Fail float3");
        }

        delete[] fa;
        delete[] fb;
        delete[] fc;
        hipFree(d_fa);
        hipFree(d_fb);
    }

    {
        float4 *fa, *fb, *fc;
        fa = new float4[msize];
        fb = new float4[msize];
        fc = new float4[msize];

        float4 *d_fa, *d_fb;

        fillMatrix<float4>(fa, msize);

        cpuRotate<float4>(fa, fb, msize);

        hipMalloc(&d_fa, sizeof(float4) * msize);
        hipMalloc(&d_fb, sizeof(float4) * msize);

        hipMemcpy(d_fa, fa, sizeof(float4) * msize, hipMemcpyHostToDevice);
        hipMemcpy(d_fb, fb, sizeof(float4) * msize, hipMemcpyHostToDevice);

        matAcc<float4>(fa, fb, msize);

        hipLaunchKernelGGL(gMatAcc, 1, msize, 0, 0, d_fa, d_fb, msize);

        hipMemcpy(fc, d_fa, sizeof(float4) * msize, hipMemcpyDeviceToHost);

        if (!isEqual<float4>(fa, fc, msize)) {
            failed("Fail float4");
        }

        delete[] fa;
        delete[] fb;
        delete[] fc;
        hipFree(d_fa);
        hipFree(d_fb);
    }
    passed();
}
