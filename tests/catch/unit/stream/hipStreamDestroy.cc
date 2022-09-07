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
#include <chrono>
#include <hip_test_common.hh>

namespace hipStreamDestroyTests {

TEST_CASE("Unit_hipStreamDestroy_Default") {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipStreamDestroy_Negative_DoubleDestroy") {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK_ERROR(hipStreamDestroy(stream), hipErrorContextIsDestroyed);
}

TEST_CASE("Unit_hipStreamDestroy_Negative_NullStream") {
  HIP_CHECK_ERROR(hipStreamDestroy(nullptr), hipErrorInvalidResourceHandle);
}

template <size_t numDataPoints> void checkDataSet(int* deviceData) {
  HIP_CHECK(hipStreamSynchronize(nullptr));
  std::array<int, numDataPoints> hostData{};
  HIP_CHECK(
      hipMemcpy(hostData.data(), deviceData, sizeof(int) * numDataPoints, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(std::begin(hostData), std::end(hostData), [](int x) { return x == 1; }));
}

__global__ void setToOne(int* x, size_t size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x[idx] = 1;
  }
}

TEST_CASE("Unit_hipStreamDestroy_WithFinishedWork") {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr int numDataPoints = 10;
  int* deviceData{};
  HIP_CHECK(hipMalloc(&deviceData, sizeof(int) * numDataPoints));
  HIP_CHECK(hipMemset(deviceData, 0, sizeof(int) * numDataPoints));

  setToOne<<<1, numDataPoints, 0, stream>>>(deviceData, numDataPoints);
  checkDataSet<numDataPoints>(deviceData);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(deviceData));
}

// hipStreamDestroy should return immediately then clean up the resources when the stream is empty
// of work
#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */
TEST_CASE("Unit_hipStreamDestroy_WithPendingWork") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-44 - expected hipStreamDestroy to return immediately then release the resources "
      "when the queued jobs are finished");
  return;
#endif
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr int numDataPoints = 10;
  int* deviceData{};
  HIP_CHECK(hipMalloc(&deviceData, sizeof(int) * numDataPoints));
  HIP_CHECK(hipMemset(deviceData, 0, sizeof(int) * numDataPoints));

  HipTest::runKernelForDuration(std::chrono::milliseconds(500), stream);
  setToOne<<<1, numDataPoints, 0, stream>>>(deviceData, numDataPoints);
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  HIP_CHECK(hipStreamDestroy(stream));
  checkDataSet<numDataPoints>(deviceData);
  HIP_CHECK(hipFree(deviceData));
}
#endif
}  // namespace hipStreamDestroyTests
