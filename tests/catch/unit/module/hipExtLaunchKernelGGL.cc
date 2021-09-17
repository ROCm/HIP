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
/*
 * Test Scenarios
   1. Verify kernel execution time of the particular kernel
   2. Verify hipExtLaunchKernelGGL API by disabling time flag in event creation
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip/hip_ext.h"

#define FOURSEC_KERNEL 4999
#define TWOSEC_KERNEL  2999

__device__ int globalvar = 1;
__global__ void TwoSecKernel_GlobalVar(int clockrate) {
  if (globalvar == 0x2222) {
    globalvar = 0x3333;
  }
  HipTest::waitKernel(2, clockrate);
  if (globalvar != 0x3333) {
    globalvar = 0x5555;
  }
}

__global__ void FourSecKernel_GlobalVar(int clockrate) {
  if (globalvar == 1) {
    globalvar = 0x2222;
  }
  HipTest::waitKernel(4, clockrate);
  if (globalvar == 0x2222) {
    globalvar = 0x4444;
  }
}


/*
 * In this Scenario, we create events by disabling the timing flag
 * We then Launch the kernel using hipExtModuleLaunchKernel by passing
 * disabled events and try to fetch kernel execution time using
 * hipEventElapsedTime API which would fail as the flag is disabled.
 */
TEST_CASE("Unit_hipExtLaunchKernelGGL_TimeFlagDisabled") {
  hipStream_t stream;
  HIP_CHECK(hipSetDevice(0));
  float time_2sec;
  hipEvent_t  start_event, end_event;
  int clkRate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  // Event Creation and Launching kernels
  HIP_CHECK(hipEventCreateWithFlags(&start_event,
                                   hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&end_event,
                                   hipEventDisableTiming));
  HIP_CHECK(hipStreamCreate(&stream));

  hipExtLaunchKernelGGL(TwoSecKernel_GlobalVar, dim3(1), dim3(1), 0,
                        stream, start_event, end_event, 0, clkRate);
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(hipEventElapsedTime(&time_2sec, start_event, end_event)
                              != hipSuccess);

  // Destroying the events and streams
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(start_event));
  HIP_CHECK(hipEventDestroy(end_event));
}
/*
 * Launching FourSecKernel and TwoSecKernel and then we try to
 * get the event elapsed time of each kernel using the start and
 * end events.The event elapsed time should return us the kernel
 * execution time for that particular kernel
*/
TEST_CASE("Unit_hipExtLaunchKernelGGL_KernelTimeExecution") {
  hipStream_t stream;
  HIP_CHECK(hipSetDevice(0));
  hipEvent_t  start_event1, end_event1, start_event2, end_event2;
  float time_4sec, time_2sec;
  int clkRate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  // Creating streams and events
  HIP_CHECK(hipEventCreate(&start_event1));
  HIP_CHECK(hipEventCreate(&end_event1));
  HIP_CHECK(hipEventCreate(&start_event2));
  HIP_CHECK(hipEventCreate(&end_event2));
  HIP_CHECK(hipStreamCreate(&stream));

  // Launching 4sec and 2sec kernels
  hipExtLaunchKernelGGL(FourSecKernel_GlobalVar, dim3(1), dim3(1), 0,
                         stream, start_event1, end_event1, 0, clkRate);
  hipExtLaunchKernelGGL(TwoSecKernel_GlobalVar, dim3(1), dim3(1), 0,
                          stream, start_event2, end_event2, 0, clkRate);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventElapsedTime(&time_4sec, start_event1, end_event1));
  HIP_CHECK(hipEventElapsedTime(&time_2sec, start_event2, end_event2));

  INFO("Expected Vs Actual: Kernel1-<" <<  FOURSEC_KERNEL << "Vs" << time_4sec
       << "Kernel2-<" << TWOSEC_KERNEL << "Vs" << time_2sec);
  // Verifying the kernel execution time
  REQUIRE(time_4sec < static_cast<float>(FOURSEC_KERNEL));
  REQUIRE(time_2sec < static_cast<float>(TWOSEC_KERNEL));

  // Destroying streams and events
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(start_event1));
  HIP_CHECK(hipEventDestroy(end_event1));
  HIP_CHECK(hipEventDestroy(start_event2));
  HIP_CHECK(hipEventDestroy(end_event2));
}
