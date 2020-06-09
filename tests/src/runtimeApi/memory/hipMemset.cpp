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
// Simple test for memset.
// Also serves as a template for other tests.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * //Small copy
 * TEST: %t -N 10    --memsetval 0x42 --memsetD32val 0x101 --memsetD16val 0x10 --memsetD8val 0x1
 * // Oddball size
 * TEST: %t -N 10013 --memsetval 0x5a --memsetD32val 0xDEADBEEF --memsetD16val 0xDEAD --memsetD8val 0xDE
 * // Big copy
 * TEST: %t -N 256M  --memsetval 0xa6 --memsetD32val 0xCAFEBABE --memsetD16val 0xCAFE --memsetD8val 0xCA
 * HIT_END
 */
#define MAX_OFFSET 3
// To test memset on unaligned pointer
#define loop(offset, offsetMax) for (int offset = offsetMax; offset >= 0; offset --)

#include <vector>
#include "hip/hip_runtime.h"
#include "test_common.h"
enum MemsetType {
  hipMemsetTypeDefault,
  hipMemsetTypeD8,
  hipMemsetTypeD16,
  hipMemsetTypeD32
};

bool testhipMemsetSmallSize(int memsetval, int p_gpuDevice)
{
  char *A_d;
  char *A_h;
  bool testResult = true;
  for ( size_t iSize = 1; iSize < 4; iSize++ ) {
    size_t Nbytes = iSize * sizeof(char);
    HIPCHECK(hipMalloc(&A_d, Nbytes));
    A_h = reinterpret_cast<char*> (malloc(Nbytes));
    printf("testhipMemsetSmallSize N=%zu  memsetval=%2x device=%d\n",
          iSize, memsetval, p_gpuDevice);
    HIPCHECK(hipMemset(A_d, memsetval, Nbytes));
    HIPCHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

    for ( int i = 0; i < iSize; i++ ) {
      if ( A_h[i] != memsetval ) {
        testResult = false;
        printf("mismatch at index:%d computed:%02x, memsetval:%02x\n",
              i, static_cast<int> (A_h[i]), static_cast<int> (memsetval));
          break;
      }
    }
  HIPCHECK(hipFree(A_d));
  free(A_h);
  }
  return testResult;
}

template<typename T>
bool testhipMemset(T*A_h, T*A_d, T memsetval, enum MemsetType type,
                  int p_gpuDevice) {
  size_t Nbytes = N * sizeof(T);
  bool testResult = true;
  HIPCHECK(hipMalloc(&A_d, Nbytes));
  A_h = reinterpret_cast<T*> (malloc(Nbytes));
  loop(offset, MAX_OFFSET) {
    if (type == hipMemsetTypeDefault) {
      printf("testhipMemset N=%zu  memsetval=%2x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemset(A_d + offset, memsetval, N - offset));
    } else if (type == hipMemsetTypeD8) {
      printf("testhipMemsetD8 N=%zu  memsetD8val=%4x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemsetD8((hipDeviceptr_t)(A_d + offset), memsetval, N - offset));
    } else if (type == hipMemsetTypeD16) {
      printf("testhipMemsetD16 N=%zu  memsetD16val=%4x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemsetD16((hipDeviceptr_t)(A_d + offset), memsetval, N - offset));
    } else if (type == hipMemsetTypeD32) {
      printf("testhipMemsetD32 N=%zu  memsetD32val=%8x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemsetD32((hipDeviceptr_t)(A_d + offset), memsetval, N - offset));
    }
    HIPCHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost) );
    for ( int i = offset; i < N; i++ ) {
      if (A_h[i] != memsetval) {
        testResult = false;
        printf("mismatch at index:%d computed:%02x, memsetval:%02x\n",
              i, static_cast<int> (A_h[i]), static_cast<int> (memsetval));
        break;
      }
    }
  }
  HIPCHECK(hipFree(A_d));
  free(A_h);
  return testResult;
}

template<typename T>
bool testhipMemsetAsync(T*A_h, T*A_d, T memsetval, enum MemsetType type,
                       int p_gpuDevice) {
  size_t Nbytes = N * sizeof(T);
  bool testResult = true;
  HIPCHECK(hipMalloc(reinterpret_cast<void**> (&A_d), Nbytes));
  A_h = reinterpret_cast<T*> (malloc(Nbytes));
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  loop(offset, MAX_OFFSET) {
    if (type == hipMemsetTypeDefault) {
      printf("testhipMemsetAsync N=%zu  memsetval=%2x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemsetAsync(A_d+offset, memsetval, Nbytes-offset, stream));
    } else if (type == hipMemsetTypeD8) {
      printf("testhipMemsetD8Async N=%zu  memsetD8val=%2x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemsetD8Async((hipDeviceptr_t)(A_d + offset), memsetval, N - offset, stream));
    } else if (type == hipMemsetTypeD16) {
      printf("testhipMemsetD16Async N=%zu  memsetD16val=%8x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemsetD16Async((hipDeviceptr_t)(A_d + offset), memsetval, N - offset, stream));
    } else if (type == hipMemsetTypeD32) {
      printf("testhipMemsetD32Async N=%zu  memsetD32val=%8x device=%d\n",
            (N - offset), memsetval, p_gpuDevice);
      HIPCHECK(hipMemsetD32Async((hipDeviceptr_t)(A_d + offset), memsetval, N - offset, stream));
    }
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipMemcpy(A_h, reinterpret_cast<void*> (A_d), Nbytes, hipMemcpyDeviceToHost));

    for ( int i = offset; i < N; i++ ) {
      if (A_h[i] != memsetval) {
        testResult = false;
        printf("mismatch at index:%d computed:%02x\n", i, static_cast<int> (A_h[i]));
        break;
      }
    }
  }
  HIPCHECK(hipFree(reinterpret_cast<void*> (A_d)) );
  HIPCHECK(hipStreamDestroy(stream));
  free(A_h);
  return testResult;
}

bool testhipMemset2AsyncOps() {
  printf("testhipMemset2AsyncOps 2 memset operations at the same time\n");
  std::vector<float> v;
  v.resize(2048);
  float* p2, *p3;
  hipMalloc(reinterpret_cast<void**>(&p2), 4096 + 4096*2);
  p3 = p2+2048;
  hipStream_t s;
  hipStreamCreate(&s);
  hipMemsetAsync(p2, 0, 32*32*4, s);
  hipMemsetD32Async((hipDeviceptr_t)p3, 0x3fe00000, 32*32, s );
  hipStreamSynchronize(s);
  for (int i = 0; i < 256; ++i) {
    hipMemsetAsync(p2, 0, 32*32*4, s);
    hipMemsetD32Async((hipDeviceptr_t)p3, 0x3fe00000, 32*32, s );
  }
  hipStreamSynchronize(s);
  hipDeviceSynchronize();
  hipMemcpy(&v[0], p2, 1024, hipMemcpyDeviceToHost);
  hipMemcpy(&v[1024], p3, 1024, hipMemcpyDeviceToHost);
  if ((v[0] != 0) || (v[1024] != 1.75f)) {
    printf("mismatch (%f != 0) or (%f != 1.75f)\n", v[0], v[1024]);
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool testResult = true;
  char * cA_d = NULL;
  char * cA_h = NULL;
  int16_t * siA_d = NULL;
  int16_t * siA_h = NULL;
  int32_t * iA_d = NULL;
  int32_t * iA_h = NULL;
  HIPCHECK(hipSetDevice(p_gpuDevice));
  testResult &= testhipMemsetSmallSize(memsetval, p_gpuDevice);

  testResult &= testhipMemset(cA_h, cA_d, memsetval, hipMemsetTypeDefault, p_gpuDevice);
  testResult &= testhipMemset(iA_h, iA_d, memsetD32val, hipMemsetTypeD32, p_gpuDevice);
  testResult &= testhipMemset(siA_h, siA_d, memsetD16val, hipMemsetTypeD16, p_gpuDevice);
  testResult &= testhipMemset(cA_h, cA_d, memsetD8val, hipMemsetTypeD8, p_gpuDevice);

  testResult &= testhipMemsetAsync(cA_h, cA_d, memsetval, hipMemsetTypeDefault, p_gpuDevice);
  testResult &= testhipMemsetAsync(iA_h, iA_d, memsetD32val, hipMemsetTypeD32, p_gpuDevice);
  testResult &= testhipMemsetAsync(siA_h, siA_d, memsetD16val, hipMemsetTypeD16, p_gpuDevice);
  testResult &= testhipMemsetAsync(cA_h, cA_d, memsetD8val, hipMemsetTypeD8, p_gpuDevice);
  testResult &= testhipMemset2AsyncOps();
  if (testResult) passed();
  failed("Output Mismatch\n");
}
