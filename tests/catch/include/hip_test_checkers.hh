#pragma once
#include "hip_test_common.hh"

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
        assert(a == b);
        return a;
      },
      expectMatch);
}


// Setters and Memory Management

template <typename T> void setDefaultData(size_t numElements, T* A_h, T* B_h, T* C_h) {
  // Initialize the host data:
  for (size_t i = 0; i < numElements; i++) {
    if (A_h) (A_h)[i] = 3.146f + i;  // Pi
    if (B_h) (B_h)[i] = 1.618f + i;  // Phi
    if (C_h) (C_h)[i] = 0.0f + i;
  }
}

template <typename T>
bool initArraysForHost(T** A_h, T** B_h, T** C_h, size_t N, bool usePinnedHost = false) {
  size_t Nbytes = N * sizeof(T);

  if (usePinnedHost) {
    if (A_h) {
      HIPCHECK(hipHostMalloc((void**)A_h, Nbytes));
    }
    if (B_h) {
      HIPCHECK(hipHostMalloc((void**)B_h, Nbytes));
    }
    if (C_h) {
      HIPCHECK(hipHostMalloc((void**)C_h, Nbytes));
    }
  } else {
    if (A_h) {
      *A_h = (T*)malloc(Nbytes);
      REQUIRE(*A_h != NULL);
    }

    if (B_h) {
      *B_h = (T*)malloc(Nbytes);
      REQUIRE(*B_h != NULL);
    }

    if (C_h) {
      *C_h = (T*)malloc(Nbytes);
      REQUIRE(*C_h != NULL);
    }
  }

  setDefaultData(N, A_h ? *A_h : NULL, B_h ? *B_h : NULL, C_h ? *C_h : NULL);
  return true;
}

template <typename T>
bool initArrays(T** A_d, T** B_d, T** C_d, T** A_h, T** B_h, T** C_h, size_t N,
                bool usePinnedHost = false) {
  size_t Nbytes = N * sizeof(T);

  if (A_d) {
    HIPCHECK(hipMalloc(A_d, Nbytes));
  }
  if (B_d) {
    HIPCHECK(hipMalloc(B_d, Nbytes));
  }
  if (C_d) {
    HIPCHECK(hipMalloc(C_d, Nbytes));
  }

  return initArraysForHost(A_h, B_h, C_h, N, usePinnedHost);
}

template <typename T> bool freeArraysForHost(T* A_h, T* B_h, T* C_h, bool usePinnedHost) {
  if (usePinnedHost) {
    if (A_h) {
      HIPCHECK(hipHostFree(A_h));
    }
    if (B_h) {
      HIPCHECK(hipHostFree(B_h));
    }
    if (C_h) {
      HIPCHECK(hipHostFree(C_h));
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
    HIPCHECK(hipFree(A_d));
  }
  if (B_d) {
    HIPCHECK(hipFree(B_d));
  }
  if (C_d) {
    HIPCHECK(hipFree(C_d));
  }

  return freeArraysForHost(A_h, B_h, C_h, usePinnedHost);
}
}  // namespace HipTest
