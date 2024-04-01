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
 * @brief Check that querying a stream with no work returns hipSuccess
 *
 **/
TEST_CASE("Unit_hipStreamQuery_WithNoWork") {
  hipStream_t stream{nullptr};

  SECTION("Null Stream") {
    HIP_CHECK(hipStreamQuery(stream));
  }

  SECTION ("Created Stream") {
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamQuery(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * @brief Check that querying a stream with finished work returns hipSuccess
 *
 **/
TEST_CASE("Unit_hipStreamQuery_WithFinishedWork") {
  hipStream_t stream{nullptr};

  SECTION("Null Stream") {
    hip::stream::empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipStreamQuery(stream));
  }

  SECTION ("Created Stream") {
    HIP_CHECK(hipStreamCreate(&stream));
    hip::stream::empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipStreamQuery(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

#if !HT_NVIDIA
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

#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */

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
    HipTest::runKernelForDuration(std::chrono::milliseconds(500), stream);
    HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
    HIP_CHECK(hipDeviceSynchronize());

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
  HipTest::runKernelForDuration(std::chrono::milliseconds(500), hip::nullStream);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize(hip::nullStream));
}

/**
 * @brief Check that querying a stream with pending work returns hipErrorNotReady
 *
 **/
TEST_CASE("Unit_hipStreamQuery_WithPendingWork") {
  hipStream_t waitingStream{nullptr};
  HIP_CHECK(hipStreamCreate(&waitingStream));

  HipTest::runKernelForDuration(std::chrono::milliseconds(500), waitingStream);

  HIP_CHECK_ERROR(hipStreamQuery(waitingStream), hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(waitingStream));
  HIP_CHECK(hipStreamQuery(waitingStream));

  HIP_CHECK(hipStreamDestroy(waitingStream));
}
#endif
