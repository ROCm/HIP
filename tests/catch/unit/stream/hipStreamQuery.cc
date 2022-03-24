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

/**
 * @brief Check that submitting work to a stream sets the status of the nullStream to
 * hipErrorNotReady
 *
 */
TEST_CASE("Unit_hipStreamQuery_SubmitWorkOnStreamAndQueryNullStream") {
  {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipStreamQuery(hip::nullStream));
    hip::stream::waiting_kernel<<<1, 1, 0, stream>>>();
    HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

    std::thread signalingThread = hip::stream::startSignalingThread();
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
TEST_CASE("Unit_hipStreamQuery_NullStreamQuery") {
  HIP_CHECK(hipStreamQuery(hip::nullStream));
  hip::stream::waiting_kernel<<<1, 1, 0, hip::nullStream>>>();
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  std::thread signalingThread = hip::stream::startSignalingThread();
  HIP_CHECK(hipStreamSynchronize(hip::nullStream));
  signalingThread.join();
}

#if HT_NVIDIA==0
/**
 * @brief Check that submitting work to a destroyed stream sets its status as
 * hipErrorContextIsDestroyed
 *
 * Test removed for Nvidia devices because it returns unexpected error
 */
TEST_CASE("Unit_hipStreamQuery_WithDestroyedStream") {
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorContextIsDestroyed);
}

/**
 * @brief Check that submitting work to an uninitialized stream sets its status as
 * hipErrorContextIsDestroyed
 *
 * Test removed for Nvidia devices because it returns unexpected error
 */
TEST_CASE("Unit_hipStreamQuery_WithUninitializedStream") {
  hipStream_t stream{reinterpret_cast<hipStream_t>(0xFFFF)};
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorContextIsDestroyed);
}
#endif

/**
 * @brief Check that querying a stream with no work returns hipSuccess
 *
 **/
TEST_CASE("Unit_hipStreamQuery_WithNoWork") {
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamQuery(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * @brief Check that querying a stream with pending work returns hipErrorNotReady
 *
 **/
TEST_CASE("Unit_hipStreamQuery_WithPendingWork") {
  hipStream_t waitingStream{nullptr};
  hipStream_t writingStream{nullptr};
  HIP_CHECK(hipStreamCreate(&waitingStream));
  HIP_CHECK(hipStreamCreate(&writingStream));

  int32_t* signalPtr;
  HIP_CHECK(hipMalloc((void**)&signalPtr, sizeof(int32_t)));
  int32_t initValue = 0;

  hipMemcpy(signalPtr, &initValue, sizeof(int32_t), hipMemcpyHostToDevice);

  // waiting kernel
  hipLaunchKernelGGL(hip::stream::waiting_kernel, dim3(1), dim3(1), 0, waitingStream, signalPtr);
  HIP_CHECK_ERROR(hipStreamQuery(waitingStream), hipErrorNotReady);
  // signaling kernel
  hipLaunchKernelGGL(hip::stream::signaling_kernel, dim3(1), dim3(1), 0, writingStream, signalPtr);

  HIP_CHECK(hipStreamSynchronize(writingStream));
  HIP_CHECK(hipStreamSynchronize(waitingStream));
  HIP_CHECK(hipStreamQuery(waitingStream));

  HIP_CHECK(hipFree(signalPtr));
  HIP_CHECK(hipStreamDestroy(writingStream));
  HIP_CHECK(hipStreamDestroy(waitingStream));
}

/**
 * @brief Empty kernel to ensure work finishes on the stream quickly
 *
 **/
__global__ void empty_kernel() {}

/**
 * @brief Check that querying a stream with finished work returns hipSuccess
 *
 **/
TEST_CASE("Unit_hipStreamQuery_WithFinishedWork") {
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));

  hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipStreamQuery(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}
