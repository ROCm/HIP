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
#include <catch.hpp>
#include <atomic>
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <cstdlib>

#define HIP_PRINT_STATUS(status) INFO(hipGetErrorName(status) << " at line: " << __LINE__);

// Not thread-safe
#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      INFO("Error: " << hipGetErrorString(localError) << "\n    Code: " << localError              \
                     << "\n    Str: " << #error << "\n    In File: " << __FILE__                   \
                     << "\n    At line: " << __LINE__);                                            \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

// Threaded HIP_CHECKs
#define HIP_CHECK_THREAD(error)                                                                    \
  {                                                                                                \
    /*To see if error has occured in previous threads, stop execution */                           \
    if (TestContext::get().hasErrorOccured() == true) {                                            \
      return; /*This will only work with std::thread and not with std::async*/                     \
    }                                                                                              \
    auto localError = error;                                                                       \
    HCResult result(__LINE__, __FILE__, localError, #error);                                       \
    TestContext::get().addResults(result);                                                         \
  }

#define REQUIRE_THREAD(condition)                                                                  \
  {                                                                                                \
    /*To see if error has occured in previous threads, stop execution */                           \
    if (TestContext::get().hasErrorOccured() == true) {                                            \
      return; /*This will only work with std::thread and not with std::async*/                     \
    }                                                                                              \
    auto localResult = (condition);                                                                \
    HCResult result(__LINE__, __FILE__, hipSuccess, #condition, localResult);                      \
    TestContext::get().addResults(result);                                                         \
  }

// Do not call before all threads have joined
#define HIP_CHECK_THREAD_FINALIZE()                                                                \
  { TestContext::get().finalizeResults(); }


// Check that an expression, errorExpr, evaluates to the expected error_t, expectedError.
#define HIP_CHECK_ERROR(errorExpr, expectedError)                                                  \
  {                                                                                                \
    hipError_t localError = errorExpr;                                                             \
    INFO("Matching Errors: "                                                                       \
         << "\n    Expected Error: " << hipGetErrorString(expectedError)                           \
         << "\n    Expected Code: " << expectedError << '\n'                                       \
         << "                  Actual Error:   " << hipGetErrorString(localError)                  \
         << "\n    Actual Code:   " << localError << "\nStr: " << #errorExpr                       \
         << "\n    In File: " << __FILE__ << "\n    At line: " << __LINE__);                       \
    REQUIRE(localError == expectedError);                                                          \
  }

// Not thread-safe
#define HIPRTC_CHECK(error)                                                                        \
  {                                                                                                \
    auto localError = error;                                                                       \
    if (localError != HIPRTC_SUCCESS) {                                                            \
      INFO("Error: " << hiprtcGetErrorString(localError) << "\n    Code: " << localError           \
                     << "\n    Str: " << #error << "\n    In File: " << __FILE__                   \
                     << "\n    At line: " << __LINE__);                                            \
      REQUIRE(false);                                                                              \
    }                                                                                              \
  }

// Although its assert, it will be evaluated at runtime
#define HIP_ASSERT(x)                                                                              \
  { REQUIRE((x)); }

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
  if (!(condition)) {                                                                              \
    printf("assertion %s at %s:%d \n", #condition, __FILE__, __LINE__);                            \
    abort();                                                                                       \
  }

#if HT_NVIDIA
#define CTX_CREATE()                                                                               \
  hipCtx_t context;                                                                                \
  initHipCtx(&context);
#define CTX_DESTROY() HIPCHECK(hipCtxDestroy(context));
#define ARRAY_DESTROY(array) HIPCHECK(hipArrayDestroy(array));
#define HIP_TEX_REFERENCE hipTexRef
#define HIP_ARRAY hiparray
static void initHipCtx(hipCtx_t* pcontext) {
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
  int device{0};
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));

  unsigned blocks = props.multiProcessorCount * blocksPerCU;
  if (blocks * threadsPerBlock < N) {
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  }

  return blocks;
}

// Threaded version of setNumBlocks - to be used in multi threaded test
// Why? because catch2 does not support multithreaded macro calls
// Make sure you call HIP_CHECK_THREAD_FINALIZE after your threads join
// Also you can not return in threaded functions, due to how HIP_CHECK_THREAD works
static inline void setNumBlocksThread(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N,
                                      unsigned& blocks) {
  int device{0};
  blocks = 0;  // incase error has occured in some other thread and the next call might not execute,
               // we set the blocks size to 0
  HIP_CHECK_THREAD(hipGetDevice(&device));
  hipDeviceProp_t props{};
  HIP_CHECK_THREAD(hipGetDeviceProperties(&props, device));

  blocks = props.multiProcessorCount * blocksPerCU;
  if (blocks * threadsPerBlock > N) {
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  }
}

static inline int RAND_R(unsigned* rand_seed) {
#if defined(_WIN32) || defined(_WIN64)
  srand(*rand_seed);
  return rand();
#else
  return rand_r(rand_seed);
#endif
}

inline bool isImageSupported() {
  int imageSupport = 1;
#if HT_AMD
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIPCHECK(hipDeviceGetAttribute(&imageSupport, hipDeviceAttributeImageSupport, device));
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
HIP_CHECK(hipGetLastError());
}

//---
struct Pinned {
  static const bool isPinned = true;
  static const char* str() { return "Pinned"; };

  static void* Alloc(size_t sizeBytes) {
    void* p;
    HIPCHECK(hipHostMalloc((void**)&p, sizeBytes));
    return p;
  };
};


//---
struct Unpinned {
  static const bool isPinned = false;
  static const char* str() { return "Unpinned"; };

  static void* Alloc(size_t sizeBytes) {
    void* p = malloc(sizeBytes);
    HIPASSERT(p);
    return p;
  };
};


struct Memcpy {
  static const char* str() { return "Memcpy"; };
};

struct MemcpyAsync {
  static const char* str() { return "MemcpyAsync"; };
};


template <typename C> struct MemTraits;


template <> struct MemTraits<Memcpy> {
  static void Copy(void* dest, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                   hipStream_t stream) {
    (void)stream;
    HIPCHECK(hipMemcpy(dest, src, sizeBytes, kind));
  }
};


template <> struct MemTraits<MemcpyAsync> {
  static void Copy(void* dest, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                   hipStream_t stream) {
    HIPCHECK(hipMemcpyAsync(dest, src, sizeBytes, kind, stream));
  }
};


namespace {
static __global__ void waitKernel(clock_t offset) {
  auto start = clock();
  while ((clock() - start) < offset) {
  }
}

// helper function used to set the device frequency variable
// estimates the number of clock ticks in 1 second
static size_t findTicksPerSecond() {
  // first read the reported clockRate as a starting point
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  clock_t devFreq = static_cast<clock_t>(prop.clockRate);  // in kHz
  clock_t clockTicksPerSecond = devFreq * 1000;

  // init
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // Warmup
  hipLaunchKernelGGL(waitKernel, dim3(1), dim3(1), 0, 0, clockTicksPerSecond);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // try 10 times to find device frequency
  // after 10 attempts the result is likely good enough so just accept it
  for (int attempts = 10; attempts > 0; --attempts) {
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(waitKernel, dim3(1), dim3(1), 0, 0, clockTicksPerSecond);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventSynchronize(stop));

    float executionTimeMs = 0;
    HIP_CHECK(hipEventElapsedTime(&executionTimeMs, start, stop));

    constexpr float tolerance = 20;
    if (fabs(executionTimeMs - 1000) <= tolerance) {
      // Timing is within accepted tolerance, break here
      break;
    } else {
      clockTicksPerSecond = (clockTicksPerSecond * 1000) / executionTimeMs;
      --attempts;
    }
  }

  // deinit
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return clockTicksPerSecond;
}
}  // namespace

// Launches a kernel which runs for specified amount of time
// Note: The current implementation uses HIP_CHECK which is not thread safe!
// Note: the function assumes execution on a single device and caches the number of clock ticks per
// second
static inline void runKernelForDuration(std::chrono::milliseconds duration,
                                        hipStream_t stream = nullptr) {
  // number of clocks the device is running at (device frequency)
  // each translation unit will have a copy of ticksPerSecond but this function isn't designed for
  // precision so that's acceptable.
  static size_t ticksPerSecond = findTicksPerSecond();
  const auto millis = duration.count();
  hipLaunchKernelGGL(waitKernel, dim3(1), dim3(1), 0, stream, ticksPerSecond * millis / 1000);
}

}  // namespace HipTest

// This must be called in the beginning of image test app's main() to indicate whether image
// is supported.
#define CHECK_IMAGE_SUPPORT                                                                        \
  if (!HipTest::isImageSupported()) {                                                              \
    INFO("Texture is not support on the device. Skipped.");                                        \
    return;                                                                                        \
  }
