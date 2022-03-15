/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include "hip_test_context.hh"
#include <catch.hpp>
#include <stdlib.h>

#define HIP_PRINT_STATUS(status) INFO(hipGetErrorName(status) << " at line: " << __LINE__);

#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      INFO("Error: " << hipGetErrorString(localError) << " Code: " << localError << " Str: "       \
                     << #error << " In File: " << __FILE__ << " At line: " << __LINE__);           \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

// Check that an expression, errorExpr, evaluates to the expected error_t, expectedError.
#define HIP_CHECK_ERROR(errorExpr, expectedError)                                                  \
  {                                                                                                \
    hipError_t localError = errorExpr;                                                             \
    INFO("Matching Errors: "                                                                       \
         << " Expected Error: " << hipGetErrorString(expectedError)                                \
         << " Expected Code: " << expectedError << '\n'                                            \
         << "                  Actual Error:   " << hipGetErrorString(localError)                  \
         << " Actual Code:   " << localError << "\nStr: " << #errorExpr                            \
         << "\nIn File: " << __FILE__ << " At line: " << __LINE__);                                \
    REQUIRE(localError == expectedError);                                                          \
  }

#define HIPRTC_CHECK(error)                                                                        \
  {                                                                                                \
    auto localError = error;                                                                       \
    if (localError != HIPRTC_SUCCESS) {                                                            \
      INFO("Error: " << hiprtcGetErrorString(localError) << " Code: " << localError << " Str: "    \
                     << #error << " In File: " << __FILE__ << " At line: " << __LINE__);           \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }
// Although its assert, it will be evaluated at runtime
#define HIP_ASSERT(x)                                                                              \
  { REQUIRE((x)); }

#ifdef __cplusplus
  #include <iostream>
  #include <iomanip>
  #include <chrono>
#endif

#define HIPCHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {      \
            printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError),            \
                   localError, #error, __FILE__, __LINE__);                                        \
            abort();                                                                               \
        }                                                                                          \
    }

#define HIPASSERT(condition)                                                                       \
    if (!(condition)) {                                                                            \
        printf("assertion %s at %s:%d \n", #condition, __FILE__, __LINE__);                        \
        abort();                                                                                   \
    }



// Utility Functions
namespace HipTest {
static inline int getDeviceCount() {
  int dev = 0;
  HIP_CHECK(hipGetDeviceCount(&dev));
  return dev;
}

// Returns the current system time in microseconds
static inline long long get_time() {
  return std::chrono::high_resolution_clock::now().time_since_epoch()
      /std::chrono::microseconds(1);
}

static inline double elapsed_time(long long startTimeUs, long long stopTimeUs) {
  return ((double)(stopTimeUs - startTimeUs)) / ((double)(1000));
}

static inline unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N) {
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

static inline int RAND_R(unsigned* rand_seed)
{
  #if defined(_WIN32) || defined(_WIN64)
        srand(*rand_seed);
        return rand();
  #else
      return rand_r(rand_seed);
  #endif
}
}
