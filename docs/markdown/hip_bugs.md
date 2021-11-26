# HIP Bugs

<!-- toc -->
- [HIP is more restrictive in enforcing restrictions](#hip-is-more-restrictive-in-enforcing-restrictions)

<!-- tocstop -->



### HIP is more restrictive in enforcing restrictions
The language specification for HIP and CUDA forbid calling a
`__device__` function in a `__host__` context. In practice, you may observe
differences in the strictness of this restriction, with HIP exhibiting a tighter
adherence to the specification and thus less tolerant of infringing code. The
solution is to ensure that all functions which are called in a
`__device__` context are correctly annotated to reflect it.

The following is an example of codes using the specification,
```
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
    ...
    passed();
}
```
For more details for the complete program, please refer to HIP test application at the link, https://github.com/ROCm-Developer-Tools/HIP/blob/main/tests/src/deviceLib/hip_floatnTM.cpp
