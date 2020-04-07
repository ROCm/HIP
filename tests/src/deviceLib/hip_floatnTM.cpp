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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc HIPCC_OPTIONS -std=c++14
 * TEST: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include <type_traits>
#include <random>
#include "test_common.h"

static std::random_device dev;
static std::mt19937 rng(dev());

template <typename T, typename M>
__host__ __device__ inline constexpr int count() {
    return sizeof(T) / sizeof(M);
}

inline float getRandomFloat(float min = 10, float max = 100) {
    std::uniform_real_distribution<float> gen(min, max);
    return gen(rng);
}

template <typename T, typename B>
void fillMatrix(T* a, int size) {
    for (int i = 0; i < size; i++) {
        T t;
        t.x = getRandomFloat();
        if constexpr (count<T, B>() >= 2) t.y = getRandomFloat();
        if constexpr (count<T, B>() >= 3) t.z = getRandomFloat();
        if constexpr (count<T, B>() >= 4) t.w = getRandomFloat();

        a[i] = t;
    }
}

// Test operations
template <typename T, typename B>
__host__ __device__ void testOperations(T& a, T& b) {
    a.x += b.x;
    a.x++;
    b.x++;
    if constexpr (count<T, B>() >= 2) {
        a.y = b.x;
        a.x = b.y;
    }
    if constexpr (count<T, B>() >= 3) {
        if (a.x > 0) b.x /= a.x;
        a.x *= b.z;
        a.y--;
    }
    if constexpr (count<T, B>() >= 4) {
        b.w = a.x;
        a.w += (-b.y);
    }
}

template <typename T, typename B>
__global__ void testOperationsGPU(T* d_a, T* d_b, int size) {
    int id = threadIdx.x;
    if (id > size) return;
    T &a = d_a[id];
    T &b = d_b[id];

    testOperations<T, B>(a, b);
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

// Main function that tests type
// T = what you want to test
// D = pack of 1 i.e. float1 int1
template <typename T, typename D>
void testType(int msize) {
    T *fa, *fb, *fc, *h_fa, *h_fb;
    fa = new T[msize];
    fb = new T[msize];
    fc = new T[msize];
    h_fa = new T[msize];
    h_fb = new T[msize];

    T *d_fa, *d_fb;

    constexpr int c = count<T, D>();

    if (c <= 0 || c >= 5) {
        failed("Invalid Size\n");
    }

    fillMatrix<T, D>(fa, msize);
    dcopy(fb, fa, msize);
    dcopy(h_fa, fa, msize);
    dcopy(h_fb, fa, msize);
    for (int i = 0; i < msize; i++) testOperations<T, D>(h_fa[i], h_fb[i]);

    hipMalloc(&d_fa, sizeof(T) * msize);
    hipMalloc(&d_fb, sizeof(T) * msize);

    hipMemcpy(d_fa, fa, sizeof(T) * msize, hipMemcpyHostToDevice);
    hipMemcpy(d_fb, fb, sizeof(T) * msize, hipMemcpyHostToDevice);

    auto kernel = testOperationsGPU<T, D>;
    hipLaunchKernelGGL(kernel, 1, msize, 0, 0, d_fa, d_fb, msize);

    hipMemcpy(fc, d_fa, sizeof(T) * msize, hipMemcpyDeviceToHost);

    bool pass = true;
    if (!isEqual<T>(h_fa, fc, msize)) {
        pass = false;
    }

    delete[] fa;
    delete[] fb;
    delete[] fc;
    delete[] h_fa;
    delete[] h_fb;
    hipFree(d_fa);
    hipFree(d_fb);

    if (!pass) {
        failed("Failed");
    }
}

int main() {
    const int msize = 100;
    // double
    testType<double1, double1>(msize);
    testType<double2, double1>(msize);
    testType<double3, double1>(msize);
    testType<double4, double1>(msize);

    // floats
    testType<float1, float1>(msize);
    testType<float2, float1>(msize);
    testType<float3, float1>(msize);
    testType<float4, float1>(msize);

    // ints
    testType<int1, int1>(msize);
    testType<int2, int1>(msize);
    testType<int3, int1>(msize);
    testType<int4, int1>(msize);

    // chars
    testType<char1, char1>(msize);
    testType<char2, char1>(msize);
    testType<char3, char1>(msize);
    testType<char4, char1>(msize);

    // long
    testType<long1, long1>(msize);
    testType<long2, long1>(msize);
    testType<long3, long1>(msize);
    testType<long4, long1>(msize);

    // longlong
    testType<longlong1, longlong1>(msize);
    testType<longlong2, longlong1>(msize);
    testType<longlong3, longlong1>(msize);
    testType<longlong4, longlong1>(msize);

    // short
    testType<short1, short1>(msize);
    testType<short2, short1>(msize);
    testType<short3, short1>(msize);
    testType<short4, short1>(msize);

    // uints
    testType<uint1, uint1>(msize);
    testType<uint2, uint1>(msize);
    testType<uint3, uint1>(msize);
    testType<uint4, uint1>(msize);

    // uchars
    testType<uchar1, uchar1>(msize);
    testType<uchar2, uchar1>(msize);
    testType<uchar3, uchar1>(msize);
    testType<uchar4, uchar1>(msize);

    // ulong
    testType<ulong1, ulong1>(msize);
    testType<ulong2, ulong1>(msize);
    testType<ulong3, ulong1>(msize);
    testType<ulong4, ulong1>(msize);

    // ulonglong
    testType<ulonglong1, ulonglong1>(msize);
    testType<ulonglong2, ulonglong1>(msize);
    testType<ulonglong3, ulonglong1>(msize);
    testType<ulonglong4, ulonglong1>(msize);

    // ushort
    testType<ushort1, ushort1>(msize);
    testType<ushort2, ushort1>(msize);
    testType<ushort3, ushort1>(msize);
    testType<ushort4, ushort1>(msize);

    passed();
}
