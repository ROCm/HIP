/*
   Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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
 * 1. Verify hipExtLaunchKernelGGL API with concurrency flag
      Verify hipExtLaunchKernelGGL API by disabling concurrency flag
   2. Verify kernel execution time of the particular kernel
   3. Verify hipExtLaunchKernelGGL API by disabling time flag in event creation
   Testcase 1 is not included now as the firmware does not support concurrency
   in the same stream.
 */
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipExtLaunchKernelGGL_KernelExeTime --tests 2 EXCLUDE_HIP_PLATFORM nvidia
 * TEST_NAMED: %t hipExtLaunchKernelGGL_TimeFlagDisabled --tests 3 EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */

#include "test_common.h"
#include "hip/hip_ext.h"
#define FOURSEC_KERNEL 4999
#define TWOSEC_KERNEL  2999

__device__ int globalvar = 1;
__global__ void TwoSecKernel(int clockrate) {
  if (globalvar == 0x2222) {
    globalvar = 0x3333;
  }
  uint64_t wait_t = 2000,
  start = clock64()/clockrate, cur;
  do { cur = (clock64()/clockrate)-start;}while (cur < wait_t);
  if (globalvar != 0x3333) {
    globalvar = 0x5555;
  }
}

__global__ void FourSecKernel(int clockrate) {
  if (globalvar == 1) {
    globalvar = 0x2222;
  }
  uint64_t wait_t = 4000,
  start = clock64()/clockrate, cur;
  do { cur = (clock64()/clockrate)-start;}while (cur < wait_t);
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
bool DisableTimeFlag() {
  bool testStatus = true;
  hipStream_t stream1;
  HIPCHECK(hipSetDevice(0));
  hipError_t e;
  float time_4sec, time_2sec;
  hipEvent_t  start_event1, end_event1;
  int clkRate = 0;
  HIPCHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
  HIPCHECK(hipEventCreateWithFlags(&start_event1,
                                   hipEventDisableTiming));
  HIPCHECK(hipEventCreateWithFlags(&end_event1,
                                   hipEventDisableTiming));
  HIPCHECK(hipStreamCreate(&stream1));
  hipExtLaunchKernelGGL((TwoSecKernel), dim3(1), dim3(1), 0,
      stream1, start_event1, end_event1, 0, clkRate);
  HIPCHECK(hipStreamSynchronize(stream1));
  e = hipEventElapsedTime(&time_2sec, start_event1, end_event1);
  if (e == hipErrorInvalidHandle) {
    testStatus = true;
  } else {
    printf("Event elapsed time is success when time flag is disabled \n");
    testStatus = false;
  }
  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipEventDestroy(start_event1));
  HIPCHECK(hipEventDestroy(end_event1));
  return testStatus;
}
/*
  In this scenario , we initially create a global device variable
 * with initial value as 1. We then launch the four sec and two sec kernels and
 * try to modify the variable.
 * In case of concurrency,the variable gets updated in four sec kernel to 0x2222
 * and then the two sec kernel would be launched parallely which would again
 * modify the global variable to 0x3333
 * In case of non concurrency,the variale gets updated in four sec kernel
 * and then in two sec kernel and the value of global variable would be 0x5555
 */
bool ConcurencyCheck_GlobalVar(int conc_flag) {
  bool testStatus = true;
  hipStream_t stream1;
  int deviceGlobal_h = 0;
  HIPCHECK(hipSetDevice(0));
  int clkRate = 0;
  HIPCHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
  HIPCHECK(hipStreamCreate(&stream1));
  hipExtLaunchKernelGGL((FourSecKernel), dim3(1), dim3(1), 0,
                         stream1, nullptr, nullptr, conc_flag, clkRate);
  hipExtLaunchKernelGGL((TwoSecKernel), dim3(1), dim3(1), 0,
                         stream1, nullptr, nullptr, conc_flag, clkRate);
  HIPCHECK(hipStreamSynchronize(stream1));
  HIPCHECK(hipMemcpyFromSymbol(&deviceGlobal_h, globalvar,
           sizeof(int)));

  if (conc_flag && deviceGlobal_h != 0x5555) {
    testStatus = true;
  } else if (!conc_flag && deviceGlobal_h == 0x5555) {
    testStatus = true;
  } else {
    printf("Concurrency check failed when conc_flag is %d ", conc_flag);
    testStatus = false;
  }
  HIPCHECK(hipStreamDestroy(stream1));
  return testStatus;
}
/*
 * Launching FourSecKernel and TwoSecKernel and then we try to
 * get the event elapsed time of each kernel using the start and
 * end events.The event elapsed time should return us the kernel
 * execution time for that particular kernel
*/
bool KernelTimeExecution() {
  bool testStatus = true;
  hipStream_t stream1;
  hipError_t e;
  HIPCHECK(hipSetDevice(0));
  hipEvent_t  start_event1, end_event1, start_event2, end_event2;
  float time_4sec, time_2sec;
  int clkRate = 0;
  HIPCHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

  HIPCHECK(hipEventCreate(&start_event1));
  HIPCHECK(hipEventCreate(&end_event1));
  HIPCHECK(hipEventCreate(&start_event2));
  HIPCHECK(hipEventCreate(&end_event2));
  HIPCHECK(hipStreamCreate(&stream1));
  hipExtLaunchKernelGGL((FourSecKernel), dim3(1), dim3(1), 0,
                         stream1, start_event1, end_event1, 0, clkRate);
  hipExtLaunchKernelGGL((TwoSecKernel), dim3(1), dim3(1), 0,
                          stream1, start_event2, end_event2, 0, clkRate);
  HIPCHECK(hipStreamSynchronize(stream1));
  e = hipEventElapsedTime(&time_4sec, start_event1, end_event1);
  e = hipEventElapsedTime(&time_2sec, start_event2, end_event2);

  if ( (time_4sec < static_cast<float>(FOURSEC_KERNEL)) &&
       (time_2sec < static_cast<float>(TWOSEC_KERNEL))) {
    testStatus = true;
  } else {
    printf("Expected Vs Actual: Kernel1-<%d Vs %f Kernel2-<%d Vs %f\n",
            FOURSEC_KERNEL, time_4sec, TWOSEC_KERNEL, time_2sec);
    testStatus = false;
  }

  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipEventDestroy(start_event1));
  HIPCHECK(hipEventDestroy(end_event1));
  HIPCHECK(hipEventDestroy(start_event2));
  HIPCHECK(hipEventDestroy(end_event2));

  return testStatus;
}

int main(int argc, char* argv[]) {
  bool testStatus = true;
  HipTest::parseStandardArguments(argc, argv, false);
  if (p_tests == 1) {
    testStatus &= ConcurencyCheck_GlobalVar(1);
    testStatus &= ConcurencyCheck_GlobalVar(0);
  } else if (p_tests == 2) {
    testStatus &= KernelTimeExecution();
  } else if (p_tests == 3) {
    testStatus &= DisableTimeFlag();
  } else {
    failed("Didnt receive any valid option.\n");
  }
  if (testStatus) {
    passed();
  } else {
    failed("Test Failed!");
  }
}
