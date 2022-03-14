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