/*
Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip_test_rtc.hh>
#include <catch.hpp>
#include <stdlib.h>

#define HIP_PRINT_STATUS(status) INFO(hipGetErrorName(status) << " at line: " << __LINE__);

// Not thread-safe
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

// Not thread-safe
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
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError), localError,      \
             #error, __FILE__, __LINE__);                                                          \
      abort();                                                                                     \
    }                                                                                              \
  }

#define HIPASSERT(condition)                                                                       \
    if (!(condition)) {                                                                            \
        printf("assertion %s at %s:%d \n", #condition, __FILE__, __LINE__);                        \
        abort();                                                                                   \
    }
#if HT_NVIDIA
#define CTX_CREATE() \
  hipCtx_t context;\
  initHipCtx(&context);
#define CTX_DESTROY() HIPCHECK(hipCtxDestroy(context));
#define ARRAY_DESTROY(array) HIPCHECK(hipArrayDestroy(array));
#define HIP_TEX_REFERENCE hipTexRef
#define HIP_ARRAY hiparray
static void initHipCtx(hipCtx_t *pcontext) {
  HIPCHECK(hipInit(0));
  hipDevice_t device;
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipCtxCreate(pcontext, 0, device));
}
#else
#define CTX_CREATE()
#define CTX_DESTROY()
#define ARRAY_DESTROY(array) HIPCHECK(hipFreeArray(array));
#define HIP_TEX_REFERENCE textureReference*
#define HIP_ARRAY hipArray*
#endif


// Utility Functions
namespace HipTest {
static inline int getDeviceCount() {
  int dev = 0;
  HIP_CHECK(hipGetDeviceCount(&dev));
  return dev;
}

// Returns the current system time in microseconds
static inline long long get_time() {
  return std::chrono::high_resolution_clock::now().time_since_epoch() /
      std::chrono::microseconds(1);
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

inline bool isImageSupported() {
    int imageSupport = 1;
#ifdef __HIP_PLATFORM_AMD__
    int device;
    HIP_CHECK(hipGetDevice(&device));
    HIPCHECK(hipDeviceGetAttribute(&imageSupport, hipDeviceAttributeImageSupport,
                                   device));
#endif
  return imageSupport != 0;
}

/**
 * Causes the test to stop and be skipped at runtime.
 * reason: Message describing the reason the test has been skipped.
 */
static inline void HIP_SKIP_TEST(char const* const reason) noexcept {
  // ctest is setup to parse for "HIP_SKIP_THIS_TEST", at which point it will skip the test.
  std::cout << "Skipping test. Reason: " << reason << '\n' << "HIP_SKIP_THIS_TEST" << std::endl;
}

/**
 * @brief Helper template that returns the expected arguments of a kernel.
 *
 * @return constexpr std::tuple<FArgs...> the expected arguments of the kernel.
 */
template <typename... FArgs> std::tuple<FArgs...> getExpectedArgs(void(FArgs...)){};

/**
 * @brief Asserts that the types of the arguments of a function match exactly with the types in the
 * function signature.
 * This is necessary because HIP RTC does not do implicit casting of the kernel
 * parameters.
 * In order to get the kernel function signature, this function should only called when
 * RTC is disabled.
 *
 * @tparam F the kernel function
 * @tparam Args the parameters that will be passed to the kernel.
 */
template <typename F, typename... Args> void validateArguments(F f, Args...) {
  using expectedArgsTuple = decltype(getExpectedArgs(f));
  static_assert(std::is_same<expectedArgsTuple, std::tuple<Args...>>::value,
                "Kernel arguments types must match exactly!");
}

/**
 * @brief Launch a kernel using either HIP or HIP RTC.
 *
 * @tparam Typenames A list of typenames used by the kernel (unused if the kernel is not a
 * template).
 * @tparam K The kernel type. Expects a function or template when RTC is disabled. Expects a
 * function pointer instead when RTC is enabled.
 * @tparam Dim Can be either dim3 or int.
 * @tparam Args A list of kernel arguments to be forwarded.
 * @param kernel The kernel to be launched (defined in kernels.hh)
 * @param numBlocks
 * @param numThreads
 * @param memPerBlock
 * @param stream
 * @param packedArgs A list of kernel arguments to be forwarded.
 */
template <typename... Typenames, typename K, typename Dim, typename... Args>
void launchKernel(K kernel, Dim numBlocks, Dim numThreads, std::uint32_t memPerBlock,
                  hipStream_t stream, Args&&... packedArgs) {
#ifndef RTC_TESTING
    validateArguments(kernel, packedArgs...);
    kernel<<<numBlocks, numThreads, memPerBlock, stream>>>(std::forward<Args>(packedArgs)...);
#else
  launchRTCKernel<Typenames...>(kernel, numBlocks, numThreads, memPerBlock, stream,
                                std::forward<Args>(packedArgs)...);
#endif
}
}  // namespace HipTest


// This must be called in the beginning of image test app's main() to indicate whether image
// is supported.
#define checkImageSupport()                                                                \
    if (!HipTest::isImageSupported())                                                      \
        { printf("Texture is not support on the device. Skipped.\n"); return; }
