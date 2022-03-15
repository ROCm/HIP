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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
namespace hipStreamCreateWithFlagsTests {

TEST_CASE("Unit_hipStreamCreateWithFlags_NullStream") {
  HIP_CHECK_ERROR(hipStreamCreateWithFlags(nullptr, hipStreamDefault), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipStreamCreateWithFlags_InvalidFlag") {
  hipStream_t stream{};
  unsigned int flag = 0xFF;
  REQUIRE(flag != hipStreamDefault);
  REQUIRE(flag != hipStreamNonBlocking);
  HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, flag), hipErrorInvalidValue);
}

template <unsigned int flagUnderTest> void happyCreateWithFlags() {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithFlags(&stream, flagUnderTest));

  unsigned int flag{};
  HIP_CHECK(hipStreamGetFlags(stream, &flag));
  REQUIRE(flag == flagUnderTest);

  int priority{};
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  // zero is considered default priority
  REQUIRE(priority == 0);

  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipStreamCreateWithFlags_Default") { happyCreateWithFlags<hipStreamDefault>(); }

TEST_CASE("Unit_hipStreamCreateWithFlags_NonBlocking") {
  happyCreateWithFlags<hipStreamNonBlocking>();
}


__global__ void setting_kernel(int* x, size_t size) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < size) {
    atomicExch(&x[tid], 1);
  }
}

__global__ void comparing_kernel(volatile int* x, size_t size) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid < size) {
    while (atomicCAS(x[tid], 1, 2)) {
    }
  }
}

// NOTE hipStreamQuery can block if called too many times
// returns true if the stream finishes before the timeout
bool waitForStream(hipStream_t stream, std::chrono::milliseconds timeout,
                   std::chrono::milliseconds queryPeriod = std::chrono::milliseconds(100)) {
  using clock = std::chrono::high_resolution_clock;
  const auto end = clock::now() + timeout;
  bool finished = hipStreamQuery(stream) == hipSuccess;
  while (!finished && clock::now() < end) {
    std::this_thread::sleep_for(queryPeriod);
    finished = hipStreamQuery(stream) == hipSuccess;
  }
  return finished;
}

void checkConcurrentSupported() {
  int device{};
  hipGetDevice(&device);
  hipDeviceProp_t deviceProps{};
  hipGetDeviceProperties(&deviceProps, device);
  REQUIRE(deviceProps.concurrentKernels);
}


TEST_CASE("Unit_hipStreamCreateWithFlags_BlockNullStream") {
  constexpr auto data_size = sizeof(int);
  hipStream_t defaultStream = GENERATE(static_cast<hipStream_t>(nullptr), hipStreamPerThread);

  int* device_data{};
  HIP_CHECK(hipMalloc(&device_data, data_size));
  HIP_CHECK(hipMemset(device_data, 0, data_size));

  hipStream_t comparingStream{};
  HIP_CHECK(hipStreamCreateWithFlags(&comparingStream, hipStreamDefault));
  hipStream_t settingStream{};
  HIP_CHECK(hipStreamCreateWithFlags(&settingStream, hipStreamNonBlocking));

  checkConcurrentSupported();

  // kernel will wait until the setting kernel sets some memory
  hipLaunchKernelGGL(comparing_kernel, dim3(1), dim3(1), 0, comparingStream, device_data, 1);
  // kernel will block, causing the first kernel to busy wait
  hipLaunchKernelGGL(setting_kernel, dim3(1), dim3(1), 0, defaultStream, device_data, 1);

  // hipStreamPerThread is non-blocking
  if (defaultStream == nullptr) {
    REQUIRE(!waitForStream(comparingStream, std::chrono::seconds(1)));

    // kernel will set some memory, allowing the first kernel to complete
    hipLaunchKernelGGL(setting_kernel, dim3(1), dim3(1), 0, settingStream, device_data, 1);
  }

  REQUIRE(waitForStream(comparingStream, std::chrono::seconds(1)));

  HIP_CHECK(hipStreamDestroy(comparingStream));
  HIP_CHECK(hipStreamDestroy(settingStream));
  HIP_CHECK(hipFree(device_data));
}

TEST_CASE("Unit_hipStreamCreateWithFlags_ConcurrentNullStream") {
  constexpr auto data_size = sizeof(int);
  hipStream_t defaultStream = GENERATE(static_cast<hipStream_t>(nullptr), hipStreamPerThread);

  int* device_data{};
  HIP_CHECK(hipMalloc(&device_data, data_size));
  HIP_CHECK(hipMemset(device_data, 0, data_size));

  hipStream_t comparingStream{};
  HIP_CHECK(hipStreamCreateWithFlags(&comparingStream, hipStreamNonBlocking));

  checkConcurrentSupported();

  // kernel will wait until the second kernel sets some memory
  hipLaunchKernelGGL(comparing_kernel, dim3(1), dim3(1), 0, comparingStream, device_data, 1);
  // kernel will set some memory, allowing the first kernel to complete
  hipLaunchKernelGGL(setting_kernel, dim3(1), dim3(1), 0, defaultStream, device_data, 1);

  REQUIRE(waitForStream(comparingStream, std::chrono::seconds(1)));

  // check that the comparing kernel did run
  int host_int = 0;
  HIP_CHECK(hipMemcpy(&host_int, device_data, data_size, hipMemcpyDeviceToHost));

  REQUIRE(host_int == 2);

  HIP_CHECK(hipStreamDestroy(comparingStream));
  HIP_CHECK(hipFree(device_data));
}
}  // namespace hipStreamCreateWithFlagsTests
