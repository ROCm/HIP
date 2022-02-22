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

namespace hipStreamSynchronizeTest {
const hipStream_t explicitStream = (hipStream_t)-1;
const hipStream_t nullStream = (hipStream_t)0;
const hipStream_t streamPerThread = (hipStream_t)2;

__device__ int defaultSemaphore = 0;

/**
 * @brief Kernel that signals a semaphore to go change value from 0 to 1.
 *
 * @param semaphore the semaphore that needs to be signaled.
 */
__global__ void signaling_kernel(int* semaphore = nullptr) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid == 0) {
    if (semaphore == nullptr) {
      atomicAdd(&defaultSemaphore, 1);
    } else {
      atomicAdd(semaphore, 1);
    }
  }
}

/**
 * @brief Kernel that busy waits until the specified semaphore goes from 0 to 1.
 *
 * @param semaphore the semaphore to wait for.
 */
__global__ void waiting_kernel(int* semaphore = nullptr) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid == 0) {
    if (semaphore == nullptr) {
      while (atomicCAS(&defaultSemaphore, 1, 2) == 0) {
      }
    } else {
      while (atomicCAS(semaphore, 1, 2) == 0) {
      }
    }
  }
}

__global__ void emptyKernel() {}

/**
 * @brief Creates a thread that runs a signaling_kernel on a non-blocking stream.
 * hipStreamNonBlocking is used here to avoid interfering with tests for the Null Stream.
 *
 * @param semaphore memory location to signal
 * @return std::thread thread that has to be joined after the testing is done.
 */
std::thread startSignalingThread(int* semaphore = nullptr) {
  std::thread signalingThread([semaphore]() {
    hipStream_t signalingStream;
    HIP_CHECK(hipStreamCreateWithFlags(&signalingStream, hipStreamNonBlocking));

    signaling_kernel<<<1, 1, 0, signalingStream>>>(semaphore);
    HIP_CHECK(hipStreamSynchronize(signalingStream));
    HIP_CHECK(hipStreamDestroy(signalingStream));
  });

  return signalingThread;
}

/**
 * @brief Check that hipStreamSynchronize handles empty streams properly.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_EmptyStream") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * @brief Check that all work executing in a stream is finished after a call to
 * hipStreamSynchronize.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_FinishWork") {
  hipStream_t stream = GENERATE_COPY(explicitStream, nullStream, streamPerThread);

  if (stream == explicitStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  waiting_kernel<<<1, 1, 0, stream>>>();
  std::thread signalingThread = startSignalingThread();
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamQuery(stream));
  signalingThread.join();

  if (stream == explicitStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * @brief Check that synchronising the Null Stream implicitly synchronises all executing streams.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_NullStreamSynchronization") {

//FIXME Report this bug to Amd
#ifdef __HIP_PLATFORM_AMD__
  int totalStreams = 2;
#else
  int totalStreams = 10;
#endif

  std::vector<hipStream_t> streams{};
  std::vector<int*> semaphores{};
  std::vector<std::thread> signalingThreads{};

  for (int i = 0; i < totalStreams; ++i) {
    hipStream_t stream;
    int* semaphore;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMalloc(&semaphore, sizeof(int)));
    HIP_CHECK(hipMemset(semaphore, 0, sizeof(int)));
    streams.push_back(stream);
    semaphores.push_back(semaphore);
  }

  for (int i = 0; i < totalStreams; ++i) {
    waiting_kernel<<<1, 1, 0, streams[i]>>>(semaphores[i]);
  }

  for (int i = 0; i < totalStreams; ++i) {
    REQUIRE(hipStreamQuery(streams[i]) == hipErrorNotReady);
  }


  for (int i = 0; i < totalStreams; ++i) {
    signalingThreads.push_back(startSignalingThread(semaphores[i]));
  }

  REQUIRE(hipStreamQuery(nullStream) == hipErrorNotReady);
  hipStreamSynchronize(nullStream);
  HIP_CHECK(hipStreamQuery(nullStream));

  for (int i = 0; i < totalStreams; ++i) {
    signalingThreads[i].join();
  }

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamQuery(streams[i]));
  }

  hipDeviceSynchronize();
  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
    HIP_CHECK(hipFree(semaphores[i]));
  }
}

/**
 * @brief Check that synchronizing one stream does implicitly synchronize other streams.
 *        Check that submiting work to the nullStream does not affect synchronization of other
 * streams. Check that querying the nullStream does not affect synchronization of other streams.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_SynchronizeStreamAndQueryNullStream") {
  hipStream_t stream1;
  hipStream_t stream2;

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  int* semaphore1;
  int* semaphore2;
  HIP_CHECK(hipMalloc(&semaphore1, sizeof(int)));
  HIP_CHECK(hipMemset(semaphore1, 0, sizeof(int)));
  HIP_CHECK(hipMalloc(&semaphore2, sizeof(int)));
  HIP_CHECK(hipMemset(semaphore2, 0, sizeof(int)));

  waiting_kernel<<<1, 1, 0, stream1>>>(semaphore1);
  waiting_kernel<<<1, 1, 0, stream2>>>(semaphore2);

  SECTION("Do Use NullStream") {}
  // FIXME Report this bug
#ifndef __HIP_PLATFORM_AMD__
  SECTION("Submit Kernel to NullStream") { emptyKernel<<<1, 1, 0, nullStream>>>(); }
  SECTION("Query NullStream") { REQUIRE(hipStreamQuery(nullStream) == hipErrorNotReady); }
#endif

  REQUIRE(hipStreamQuery(stream1) == hipErrorNotReady);
  REQUIRE(hipStreamQuery(stream2) == hipErrorNotReady);

  std::thread signalingThread = startSignalingThread(semaphore1);
  HIP_CHECK(hipStreamSynchronize(stream1));
  signalingThread.join();
  HIP_CHECK(hipStreamQuery(stream1));
  REQUIRE(hipStreamQuery(stream2) == hipErrorNotReady);
  REQUIRE(hipStreamQuery(nullStream) == hipErrorNotReady);

  std::thread signalingThread2 = startSignalingThread(semaphore2);
  signalingThread2.join();

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(semaphore1));
  HIP_CHECK(hipFree(semaphore2));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}

/**
 * @brief Check that submitting work to a stream sets the status of the nullStream to
 * hipErrorNotReady
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_SubmitWorkOnStreamAndQueryNullStream") {
  {
    // FIXME This is a hipStreamQuery test
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    REQUIRE(hipStreamQuery(nullStream) == hipSuccess);
    waiting_kernel<<<1, 1, 0, stream>>>();
    REQUIRE(hipStreamQuery(stream) == hipErrorNotReady);

    std::thread signalingThread = startSignalingThread();
    HIP_CHECK(hipDeviceSynchronize());
    signalingThread.join();
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * @brief Check that submitting work to the nullStream properly sets its status as
 * hipErrorNotReady.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_NullStreamQuery") {
  // FIXME This is a hipStreamQuery test
  HIP_CHECK(hipStreamQuery(nullStream));
  waiting_kernel<<<1, 1, 0, nullStream>>>();
  REQUIRE(hipStreamQuery(nullStream) == hipErrorNotReady);

  std::thread signalingThread = startSignalingThread();
  HIP_CHECK(hipStreamSynchronize(nullStream));
  signalingThread.join();
}

/**
 * @brief Check that synchronizing the Null stream also synchronizes the hipStreamPerThread
 * special stream.
 *
 */
TEST_CASE("Unit_hipStreamSynchronize_NullStreamAndStreamPerThread") {
  waiting_kernel<<<1, 1, 0, streamPerThread>>>();
  REQUIRE(hipStreamQuery(nullStream) == hipErrorNotReady);
  REQUIRE(hipStreamQuery(streamPerThread) == hipErrorNotReady);
  waiting_kernel<<<1, 1, 0, nullStream>>>();
  std::thread signalingThread = startSignalingThread();
  HIP_CHECK(hipStreamSynchronize(nullStream))
  REQUIRE(hipStreamQuery(streamPerThread) == hipSuccess);
  REQUIRE(hipStreamQuery(nullStream) == hipSuccess);
  signalingThread.join();
}
}  // namespace hipStreamSynchronizeTest