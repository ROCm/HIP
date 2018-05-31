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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "test_common.h"
#include <hip/hip_runtime.h>
#include <hip/math_functions.h>
#include <hip/hip_runtime_api.h>

#define N 512
#define SIZE N * sizeof(float)

__global__ void test_sincosf(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    sincosf(a[tid], b + tid, c + tid);
}

__global__ void test_sincospif(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    sincospif(a[tid], b + tid, c + tid);
}

__global__ void test_fdividef(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    c[tid] = fdividef(a[tid], b[tid]);
}

__global__ void test_llrintf(float* a, long long int* b) {
    int tid = threadIdx.x;
    b[tid] = llrintf(a[tid]);
}

__global__ void test_lrintf(float* a, long int* b) {
    int tid = threadIdx.x;
    b[tid] = lrintf(a[tid]);
}

__global__ void test_rintf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = rintf(a[tid]);
}

__global__ void test_llroundf(float* a, long long int* b) {
    int tid = threadIdx.x;
    b[tid] = llroundf(a[tid]);
}

__global__ void test_lroundf(float* a, long int* b) {
    int tid = threadIdx.x;
    b[tid] = lroundf(a[tid]);
}

__global__ void test_rhypotf(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    c[tid] = rhypotf(a[tid], b[tid]);
}

__global__ void test_norm3df(float* a, float* b, float* c, float* d) {
    int tid = threadIdx.x;
    d[tid] = norm3df(a[tid], b[tid], c[tid]);
}

__global__ void test_norm4df(float* a, float* b, float* c, float* d, float* e) {
    int tid = threadIdx.x;
    e[tid] = norm4df(a[tid], b[tid], c[tid], d[tid]);
}

__global__ void test_normf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = normf(N, a);
}

__global__ void test_rnorm3df(float* a, float* b, float* c, float* d) {
    int tid = threadIdx.x;
    d[tid] = rnorm3df(a[tid], b[tid], c[tid]);
}

__global__ void test_rnorm4df(float* a, float* b, float* c, float* d, float* e) {
    int tid = threadIdx.x;
    e[tid] = rnorm4df(a[tid], b[tid], c[tid], d[tid]);
}

__global__ void test_rnormf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = rnormf(N, a);
}

__global__ void test_erfinvf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = erff(erfinvf(a[tid]));
}


bool run_sincosf() {
    float *A, *Ad, *B, *C, *Bd, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_sincosf, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[i] == sinf(1.0f)) {
            passed = 1;
        }
    }
    passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] == cosf(1.0f)) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_sincospif() {
    float *A, *Ad, *B, *C, *Bd, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_sincospif, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[i] - sinf(3.14 * 1.0f) < 0.1) {
            passed = 1;
        }
    }
    passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] - cosf(3.14 * 1.0f) < 0.1) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_fdividef() {
    float *A, *Ad, *B, *C, *Bd, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_fdividef, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] == A[i] / B[i]) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_llrintf() {
    float *A, *Ad;
    long long int *B, *Bd;
    A = new float[N];
    B = new long long int[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, N * sizeof(long long int));
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_llrintf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, N * sizeof(long long int), hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_lrintf() {
    float *A, *Ad;
    long int *B, *Bd;
    A = new float[N];
    B = new long int[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, N * sizeof(long int));
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_lrintf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, N * sizeof(long int), hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_rintf() {
    float *A, *Ad;
    float *B, *Bd;
    A = new float[N];
    B = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_rintf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        float x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}


bool run_llroundf() {
    float *A, *Ad;
    long long int *B, *Bd;
    A = new float[N];
    B = new long long int[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, N * sizeof(long long int));
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_llroundf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, N * sizeof(long long int), hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_lroundf() {
    float *A, *Ad;
    long int *B, *Bd;
    A = new float[N];
    B = new long int[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, N * sizeof(long int));
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_lroundf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, N * sizeof(long int), hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}


bool run_norm3df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
    }
    val = sqrtf(1.0f + 4.0f + 9.0f);
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMalloc((void**)&Dd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_norm3df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);
    hipMemcpy(D, Dd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (D[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);
    hipFree(Dd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_norm4df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd, *E, *Ed;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    E = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
        D[i] = 4.0f;
    }
    val = sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMalloc((void**)&Dd, SIZE);
    hipMalloc((void**)&Ed, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Dd, D, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_norm4df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd, Ed);
    hipMemcpy(E, Ed, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (E[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);
    hipFree(Dd);
    hipFree(Ed);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_normf() {
    float *A, *Ad, *B, *Bd;
    A = new float[N];
    B = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 0.0f;
        val += 1.0f;
    }
    val = sqrtf(val);
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_normf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[0] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_rhypotf() {
    float *A, *Ad, *B, *Bd, *C, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    val = 1 / sqrtf(1.0f + 4.0f);
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_rhypotf, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_rnorm3df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
    }
    val = 1 / sqrtf(1.0f + 4.0f + 9.0f);
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMalloc((void**)&Dd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_rnorm3df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);
    hipMemcpy(D, Dd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (D[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);
    hipFree(Dd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_rnorm4df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd, *E, *Ed;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    E = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
        D[i] = 4.0f;
    }
    val = 1 / sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMalloc((void**)&Dd, SIZE);
    hipMalloc((void**)&Ed, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Dd, D, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_rnorm4df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd, Ed);
    hipMemcpy(E, Ed, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (E[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);
    hipFree(Dd);
    hipFree(Ed);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_rnormf() {
    float *A, *Ad, *B, *Bd;
    A = new float[N];
    B = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 0.0f;
        val += 1.0f;
    }
    val = 1 / sqrtf(val);
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_rnormf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[0] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

bool run_erfinvf() {
    float *A, *Ad, *B, *Bd;
    A = new float[N];
    B = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = -0.6f;
        B[i] = 0.0f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(test_erfinvf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[i] - A[i] < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    hipFree(Ad);
    hipFree(Bd);

    if (passed == 1) {
        return true;
    }
    assert(passed == 1);
    return false;
}

int main() {
    if (run_sincosf() && run_sincospif() && run_fdividef() && run_llrintf() && run_norm3df() &&
        run_norm4df() && run_normf() && run_rnorm3df() && run_rnorm4df() && run_rnormf() &&
        run_lroundf() && run_llroundf() && run_rintf() && run_rhypotf() && run_erfinvf()) {
        passed();
    }
}
