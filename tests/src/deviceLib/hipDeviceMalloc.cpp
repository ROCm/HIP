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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/* HIT_START
 * BUILD: %t %s NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM all
 * TEST: %t EXCLUDE_HIP_PLATFORM all
 * HIT_END
 */
#include "test_common.h"
#include <iostream>
#include <complex>

// Tolerance for error
const double tolerance = 1e-6;
const bool verbose = false;

#define BLKDIM_X 64
#define BLKDIM_Y 1
#define BLKDIM_Z 1
#define NUM_BLK_X 1
#define NUM_BLK_Y 1
#define NUM_BLK_Z 1

#define LEN (BLKDIM_X * BLKDIM_Y * BLKDIM_Z * NUM_BLK_X * NUM_BLK_Y * NUM_BLK_Z)

#define ALL_FUN \
  OP(add) \
  OP(sub) \
  OP(mul) \
  OP(div)

#define OP(x) CK_##x,
enum CalcKind {
  ALL_FUN
};
#undef OP

#define OP(x) case CK_##x: return #x;
std::string getName(enum CalcKind CK) {
  switch(CK){
  ALL_FUN
  }
}
#undef OP

// Calculates function.
// If the function has one argument, B is ignored.
#define ONE_ARG(func)                                                                              \
    case CK_##func:                                                                                \
        return std::func(A);

template <typename FloatT>
__device__ __host__ FloatT calc(FloatT A, FloatT B, enum CalcKind CK) {
    switch (CK) {
        case CK_add:
            return A + B;
        case CK_sub:
            return A - B;
        case CK_mul:
            return A * B;
        case CK_div:
            return A / B;
    }
}

// Allocate memory in kernel and save the address to pA and pB.
// Copy value from A, B to allocated memory.
template <typename FloatT>
__global__ void kernel_alloc(FloatT* A, FloatT* B, FloatT** pA, FloatT** pB) {
    int tx = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x
        + (hipThreadIdx_y + hipBlockDim_y * hipBlockIdx_y) * hipBlockDim_x
        + (hipThreadIdx_z + hipBlockDim_z * hipBlockIdx_z) * hipBlockDim_x
        * hipBlockDim_y;
    if (tx == 0) {
        *pA = (FloatT*)malloc(sizeof(FloatT) * LEN);
        *pB = (FloatT*)malloc(sizeof(FloatT) * LEN);
        for (int i = 0; i < LEN; i++) {
            (*pA)[i] = A[i];
            (*pB)[i] = B[i];
      }
    }
}

// Do calculation using values saved in allocated memmory. pA, pB are buffers
// containing the address of the device-side allocated array.
template <typename FloatT>
__global__ void kernel_free(FloatT** pA, FloatT** pB, FloatT* C, enum CalcKind CK) {
    int tx = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x
        + (hipThreadIdx_y + hipBlockDim_y * hipBlockIdx_y) * hipBlockDim_x
        + (hipThreadIdx_z + hipBlockDim_z * hipBlockIdx_z) * hipBlockDim_x
        * hipBlockDim_y;
    C[tx] = calc<FloatT>((*pA)[tx], (*pB)[tx], CK);
    if (tx == 0) {
        free(*pA);
        free(*pB);
    }
}

template<typename FloatT>
void test() {
    FloatT *A, *Ad, *B, *Bd, *C, *Cd, *D;
    A = new FloatT[LEN];
    B = new FloatT[LEN];
    C = new FloatT[LEN];
    D = new FloatT[LEN];
    hipMalloc((void**)&Ad, sizeof(FloatT) * LEN);
    hipMalloc((void**)&Bd, sizeof(FloatT) * LEN);
    hipMalloc((void**)&Cd, sizeof(FloatT) * LEN);

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = (i + 1) * 1.0f;
        B[i] = A[i];
        C[i] = A[i];
    }
    hipMemcpy(Ad, A, sizeof(FloatT) * LEN, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, sizeof(FloatT) * LEN, hipMemcpyHostToDevice);

    // Run kernel for a calculation kind and verify by comparing with host
    // calculation result. Returns false if fails.
    auto test_fun = [&](enum CalcKind CK) {
      // kernel_alloc allocates memory on device side and initialize it.
      // kernel_free uses allocated memory from kernel_alloc and does the
      // calculation then free the memory.
      // pA and pB are buffers to pass the device-side allocated memory address
      // from kernel_alloc to kernel_free.
      FloatT **pA, **pB;
      hipMalloc((FloatT***)&pA, sizeof(FloatT*));
      hipMalloc((FloatT***)&pB, sizeof(FloatT*));
      dim3 blkDim(BLKDIM_X, BLKDIM_Y, BLKDIM_Z);
      dim3 numBlk(NUM_BLK_X, NUM_BLK_Y, NUM_BLK_Z);
      hipLaunchKernelGGL(kernel_alloc<FloatT>, numBlk, blkDim, 0, 0,
          Ad, Bd, pA, pB);
      hipDeviceSynchronize();
      hipLaunchKernelGGL(kernel_free<FloatT>, numBlk, blkDim, 0, 0,
          pA, pB, Cd, CK);
      hipMemcpy(C, Cd, sizeof(FloatT) * LEN, hipMemcpyDeviceToHost);
      hipFree(pA);
      hipFree(pB);
      for (int i = 0; i < LEN; i++) {
          FloatT Expected = calc(A[i], B[i], CK);
          FloatT error = std::abs(C[i] - Expected);
          if (std::abs(Expected) > tolerance) error /= std::abs(Expected);
          bool pass = error < tolerance;
          if (verbose || !pass) {
              std::cout << "Function: " << getName(CK) << " Operands: " << A[i] << " " << B[i]
                        << " Result: " << C[i] << " Expected: " << Expected << " Error: " << error
                        << " Pass: " << pass << std::endl;
        }
        if (!pass)
          return false;
      }
      return true;
    };

#define OP(x) assert(test_fun(CK_##x));
    ALL_FUN
#undef OP

    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
}

int main() {
  test<float>();
  test<double>();
  passed();
  return 0;
}
