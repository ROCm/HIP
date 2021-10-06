/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include "hip_test_common.hh"
#include <iostream>
#define guarantee(cond, str)                                                                        \
   {                                                                                                \
     if (!(cond)) {                                                                                 \
       INFO("guarantee failed: " << str);                                                           \
       abort();                                                                                     \
     }                                                                                              \
   }


namespace HipTest {
template <typename T>
size_t checkVectors(T* A, T* B, T* Out, size_t N, T (*F)(T a, T b), bool expectMatch = true,
                    bool reportMismatch = true) {
  size_t mismatchCount = 0;
  size_t firstMismatch = 0;
  size_t mismatchesToPrint = 10;
  for (size_t i = 0; i < N; i++) {
    T expected = F(A[i], B[i]);
    if (Out[i] != expected) {
      if (mismatchCount == 0) {
        firstMismatch = i;
      }
      mismatchCount++;
      if ((mismatchCount <= mismatchesToPrint) && expectMatch) {
        INFO("Mismatch at " << i << " Computed: " << Out[i] << " Expeted: " << expected);
        CHECK(false);
      }
    }
  }

  if (reportMismatch) {
    if (expectMatch) {
      if (mismatchCount) {
        INFO(mismatchCount << " Mismatches  First Mismatch at index : " << firstMismatch);
        REQUIRE(false);
      }
    } else {
      if (mismatchCount == 0) {
        INFO("Expected Mismatch but not found any");
        REQUIRE(false);
      }
    }
  }

  return mismatchCount;
}
template<typename T> // pointer type
bool checkArray(T* hData, T* hOutputData, size_t width, size_t height,size_t depth = 1) {
  for (size_t i = 0; i < depth; i++) {
    for (size_t j = 0; j < height; j++) {
      for (size_t k = 0; k < width; k++) {
        int offset = i*width*height + j*width + k;
        if (hData[offset] != hOutputData[offset]) {
          INFO("Mismatch at ["  << i << "," << j << "," << k << "]:"
               << hData[offset] << "----" << hOutputData[offset]);
          CHECK(false);
          return false;
        }
      }
    }
  }
  return true;
}

template <typename T>
size_t checkVectorADD(T* A_h, T* B_h, T* result_H, size_t N, bool expectMatch = true,
                      bool reportMismatch = true) {
  return checkVectors<T>(
      A_h, B_h, result_H, N, [](T a, T b) { return a + b; }, expectMatch, reportMismatch);
}

template <typename T>
void checkTest(T* expected_H, T* result_H, size_t N, bool expectMatch = true) {
  checkVectors<T>(
      expected_H, expected_H, result_H, N,
      [](T a, T b) {
        guarantee(a == b, "Both values should be equal");
        return a;
      },
      expectMatch);
}


// Setters and Memory Management

template <typename T> void setDefaultData(size_t numElements, T* A_h, T* B_h, T* C_h) {
  // Initialize the host data:

  for (size_t i = 0; i < numElements; i++) {
    if (std::is_same<T, int>::value || std::is_same<T, unsigned int>::value) {
      if (A_h) A_h[i] = 3;
      if (B_h) B_h[i] = 4;
      if (C_h) C_h[i] = 5;
    } else if(std::is_same<T, char>::value || std::is_same<T, unsigned char>::value) {
      if (A_h) A_h[i] = 'a';
      if (B_h) B_h[i] = 'b';
      if (C_h) C_h[i] = 'c';
    } else {
      if (A_h) A_h[i] = 3.146f + i;
      if (B_h) B_h[i] = 1.618f + i;
      if (C_h) C_h[i] = 1.4f + i;
    }
  }
}

template <typename T>
bool initArraysForHost(T** A_h, T** B_h, T** C_h, size_t N, bool usePinnedHost = false) {
  size_t Nbytes = N * sizeof(T);

  if (usePinnedHost) {
    if (A_h) {
      HIP_CHECK(hipHostMalloc((void**)A_h, Nbytes));
    }
    if (B_h) {
      HIP_CHECK(hipHostMalloc((void**)B_h, Nbytes));
    }
    if (C_h) {
      HIP_CHECK(hipHostMalloc((void**)C_h, Nbytes));
    }
  } else {
    if (A_h) {
      *A_h = (T*)malloc(Nbytes);
      REQUIRE(*A_h != nullptr);
    }

    if (B_h) {
      *B_h = (T*)malloc(Nbytes);
      REQUIRE(*B_h != nullptr);
    }

    if (C_h) {
      *C_h = (T*)malloc(Nbytes);
      REQUIRE(*C_h != nullptr);
    }
  }

  setDefaultData(N, A_h ? *A_h : nullptr, B_h ? *B_h : nullptr, C_h ? *C_h : nullptr);
  return true;
}

template <typename T>
bool initArrays(T** A_d, T** B_d, T** C_d, T** A_h, T** B_h, T** C_h, size_t N,
                bool usePinnedHost = false) {
  size_t Nbytes = N * sizeof(T);

  if (A_d) {
    HIP_CHECK(hipMalloc(A_d, Nbytes));
  }
  if (B_d) {
    HIP_CHECK(hipMalloc(B_d, Nbytes));
  }
  if (C_d) {
    HIP_CHECK(hipMalloc(C_d, Nbytes));
  }

  return initArraysForHost(A_h, B_h, C_h, N, usePinnedHost);
}

template <typename T> bool freeArraysForHost(T* A_h, T* B_h, T* C_h, bool usePinnedHost) {
  if (usePinnedHost) {
    if (A_h) {
      HIP_CHECK(hipHostFree(A_h));
    }
    if (B_h) {
      HIP_CHECK(hipHostFree(B_h));
    }
    if (C_h) {
      HIP_CHECK(hipHostFree(C_h));
    }
  } else {
    if (A_h) {
      free(A_h);
    }
    if (B_h) {
      free(B_h);
    }
    if (C_h) {
      free(C_h);
    }
  }
  return true;
}

template <typename T>
bool freeArrays(T* A_d, T* B_d, T* C_d, T* A_h, T* B_h, T* C_h, bool usePinnedHost) {
  if (A_d) {
    HIP_CHECK(hipFree(A_d));
  }
  if (B_d) {
    HIP_CHECK(hipFree(B_d));
  }
  if (C_d) {
    HIP_CHECK(hipFree(C_d));
  }

  return freeArraysForHost(A_h, B_h, C_h, usePinnedHost);
}

template <typename T>
unsigned setNumBlocks(T blocksPerCU, T threadsPerBlock,
    size_t N) {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));

  unsigned blocks = props.multiProcessorCount * blocksPerCU;
  if (blocks * threadsPerBlock > N) {
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  }
  return blocks;
}

}  // namespace HipTest
