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
/*
Testcase Scenarios :
Unit_hipEventSynchronize_Default_Positive- Test synchronization of an event that is completed after a simple kernel launch
Unit_hipEventSynchronize_NoEventRecord_Positive - Test synchronization of an event that has not been recorded
*/

#include <hip_test_common.hh>

#include <kernels.hh>
#include <hip_test_checkers.hh>

TEST_CASE("Unit_hipEventSynchronize_Default_Positive") {
  constexpr size_t N = 1024;

  constexpr int blocks = 1024;

  constexpr size_t Nbytes = N * sizeof(float);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  hipEvent_t end_event;
  HIP_CHECK(hipEventCreate(&end_event));

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  HipTest::launchKernel<float>(HipTest::vectorADD<float>, blocks, 1, 0, 0,
                              static_cast<const float*>(A_d), static_cast<const float*>(B_d),
                              C_d, N);

  // Record the end_event
  HIP_CHECK(hipEventRecord(end_event, NULL));
  // Wait for the end_event to complete
  HIP_CHECK(hipEventSynchronize(end_event));

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipEventDestroy(end_event));

  HipTest::checkVectorADD(A_h, B_h, C_h, N, true);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

TEST_CASE("Unit_hipEventSynchronize_NoEventRecord_Positive") {
  constexpr size_t N = 1024;

  constexpr int blocks = 1024;

  constexpr size_t Nbytes = N * sizeof(float);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  hipEvent_t dummy_event;
  HIP_CHECK(hipEventCreate(&dummy_event));

  hipEvent_t end_event;
  HIP_CHECK(hipEventCreate(&end_event));

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  HipTest::launchKernel<float>(HipTest::vectorADD<float>, blocks, 1, 0, 0,
                              static_cast<const float*>(A_d), static_cast<const float*>(B_d),
                              C_d, N);

  // Record the end_event
  HIP_CHECK(hipEventRecord(end_event, NULL));

  // When hipEventSynchronized is called on event that has not been recorded,
  // the function returns immediately
  HIP_CHECK(hipEventSynchronize(dummy_event));

  // End event has not been completed
  HIP_CHECK_ERROR(hipEventQuery(end_event), hipErrorNotReady);
  // Wait for end_event to complete
  HIP_CHECK(hipEventSynchronize(end_event));

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipEventDestroy(dummy_event));
  HIP_CHECK(hipEventDestroy(end_event));

  HipTest::checkVectorADD(A_h, B_h, C_h, N, true);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}
