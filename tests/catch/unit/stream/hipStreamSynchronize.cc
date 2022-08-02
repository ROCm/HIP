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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include "streamCommon.hh"

namespace hipStreamSynchronizeTest {

__global__ void emptyKernel() {}

/**
 * @brief Check that hipStreamSynchronize handles empty streams properly.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_EmptyStream") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * @brief Check that all work executing in a stream is finished after a call to
 * hipStreamSynchronize.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_FinishWork") {
  const hipStream_t explicitStream = reinterpret_cast<hipStream_t>(-1);
  hipStream_t stream = GENERATE_COPY(explicitStream, hip::nullStream, hip::streamPerThread);

  if (stream == explicitStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  HipTest::runKernelForDuration(std::chrono::milliseconds(500), stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamQuery(stream));

  if (stream != hip::nullStream && stream != hip::streamPerThread) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * @brief Check that synchronizing the nullStream implicitly synchronizes all executing streams.
 */
TEST_CASE("Unit_hipStreamSynchronize_NullStreamSynchronization") {
  int totalStreams = 10;

  std::vector<hipStream_t> streams{};

  for (int i = 0; i < totalStreams; ++i) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    streams.push_back(stream);
  }

  for (int i = 0; i < totalStreams; ++i) {
    HipTest::runKernelForDuration(std::chrono::milliseconds(1000), streams[i]);
  }

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK_ERROR(hipStreamQuery(streams[i]), hipErrorNotReady);
  }

  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize(hip::nullStream));
  HIP_CHECK(hipStreamQuery(hip::nullStream));

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamQuery(streams[i]));
  }

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

/**
 * @brief Check that synchronizing one stream does implicitly synchronize other streams.
 *        Check that submiting work to the nullStream does not affect synchronization of other
 * streams. Check that querying the nullStream does not affect synchronization of other streams.
 */
TEST_CASE("Unit_hipStreamSynchronize_SynchronizeStreamAndQueryNullStream") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-22");
#else

  hipStream_t stream1;
  hipStream_t stream2;

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  HipTest::runKernelForDuration(std::chrono::milliseconds(500), stream1);
  HipTest::runKernelForDuration(std::chrono::milliseconds(2000), stream2);

  SECTION("Do not use NullStream") {}
  SECTION("Submit Kernel to NullStream") { emptyKernel<<<1, 1, 0, hip::nullStream> > >(); }
  SECTION("Query NullStream") {
    HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
  }

  HIP_CHECK_ERROR(hipStreamQuery(stream1), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);


  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamQuery(stream1));
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize(stream2));
  HIP_CHECK(hipStreamQuery(stream2));

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
#endif
}

/**
 * @brief Check that synchronizing the nullStream also synchronizes the hipStreamPerThread
 * special stream.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_NullStreamAndStreamPerThread") {
  HipTest::runKernelForDuration(std::chrono::milliseconds(500), hip::streamPerThread);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::streamPerThread), hipErrorNotReady);
  HipTest::runKernelForDuration(std::chrono::milliseconds(500), hip::nullStream);
  HIP_CHECK(hipStreamSynchronize(hip::nullStream))
  HIP_CHECK_ERROR(hipStreamQuery(hip::streamPerThread), hipSuccess);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipSuccess);
}
}  // namespace hipStreamSynchronizeTest