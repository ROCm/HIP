/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

// TODO Enable it after hipStreamAttachMemAsync is feature complete on HIP

#include <hip_test_common.hh>
#include <memory>

__device__ __managed__ int var = 0;

enum class StreamAttachTestType { NullStream = 0, StreamPerThread, CreatedStream };

TEST_CASE("Unit_hipStreamAttachMemAsync_Negative") {
  hipStream_t stream{nullptr};

  auto streamType =
      GENERATE(StreamAttachTestType::NullStream, StreamAttachTestType::StreamPerThread,
               StreamAttachTestType::CreatedStream);

  if (streamType == StreamAttachTestType::StreamPerThread) {
    stream = hipStreamPerThread;
  } else if (streamType == StreamAttachTestType::CreatedStream) {
    HIP_CHECK(hipStreamCreate(&stream));
    REQUIRE(stream != nullptr);
  }

  SECTION("Invalid Resource Handle") {
    int definitelyNotAManagedVariable = 0;
    HIP_CHECK_ERROR(
        hipStreamAttachMemAsync(stream, reinterpret_cast<void*>(&definitelyNotAManagedVariable),
                                sizeof(int), hipMemAttachSingle),
        hipErrorInvalidValue);
  }

  SECTION("Invalid devptr") {
    HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream, nullptr, sizeof(int), hipMemAttachSingle),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Resource Size") {
    HIP_CHECK_ERROR(hipStreamAttachMemAsync(stream, reinterpret_cast<void*>(&var), sizeof(int) - 1,
                                            hipMemAttachSingle),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Flags") {
    HIP_CHECK_ERROR(
        hipStreamAttachMemAsync(stream, reinterpret_cast<void*>(&var), sizeof(int) - 1,
                                hipMemAttachSingle | hipMemAttachHost | hipMemAttachGlobal),
        hipErrorInvalidValue);
  }

  if (streamType == StreamAttachTestType::CreatedStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

__global__ void kernel(int* ptr, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    ptr[i] = 1024;
  }
}

constexpr size_t size = 1024;
__device__ __managed__ int m_memory[size];

TEST_CASE("Unit_hipStreamAttachMemAsync_UseCase") {
  hipStream_t stream{nullptr};

  auto streamType =
      GENERATE(StreamAttachTestType::NullStream, StreamAttachTestType::StreamPerThread,
               StreamAttachTestType::CreatedStream);

  if (streamType == StreamAttachTestType::CreatedStream) {
    HIP_CHECK(hipStreamCreate(&stream));
    REQUIRE(stream != nullptr);
  }

  SECTION("Size zero is valid") {
    int* d_memory{nullptr};
    HIP_CHECK(hipMallocManaged(&d_memory, sizeof(int) * size, hipMemAttachHost));
    HIP_CHECK(
        hipStreamAttachMemAsync(stream, reinterpret_cast<void*>(d_memory), 0, hipMemAttachHost));
    HIP_CHECK(hipStreamSynchronize(stream));  // Wait for command to complete
    HIP_CHECK(hipFree(d_memory));
  }

  SECTION("Access from device and host") {
    int* d_memory{nullptr};

    HIP_CHECK(hipMallocManaged(&d_memory, sizeof(int) * size, hipMemAttachHost));
    HIP_CHECK(hipMemset(d_memory, 0, sizeof(int) * size));
    HIP_CHECK(
        hipStreamAttachMemAsync(stream, reinterpret_cast<void*>(d_memory), 0, hipMemAttachHost));
    HIP_CHECK(hipStreamSynchronize(stream));  // Wait for the command to complete

    kernel<<<1, size, 0, stream>>>(d_memory, size);
    HIP_CHECK(hipStreamSynchronize(stream));  // Wait for the kernel to complete

    auto ptr = std::make_unique<int[]>(size);
    std::copy(d_memory, d_memory + size, ptr.get());

    HIP_CHECK(hipFree(d_memory));

    REQUIRE(std::all_of(ptr.get(), ptr.get() + size, [](int n) { return n == size; }));
  }

  SECTION("Access ManagedMemory") {
    HIP_CHECK(hipMemset(m_memory, 0, sizeof(int) * size));
    HIP_CHECK(
        hipStreamAttachMemAsync(stream, reinterpret_cast<void*>(m_memory), 0, hipMemAttachHost));
    HIP_CHECK(hipStreamSynchronize(stream));  // Wait for the command to complete

    kernel<<<1, size, 0, stream>>>(m_memory, size);
    HIP_CHECK(hipStreamSynchronize(stream));  // Wait for the kernel to complete

    auto ptr = std::make_unique<int[]>(size);
    std::copy(m_memory, m_memory + size, ptr.get());

    REQUIRE(std::all_of(ptr.get(), ptr.get() + size, [](int n) { return n == size; }));
  }

  if (streamType == StreamAttachTestType::CreatedStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}