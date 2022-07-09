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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip/hip_ext.h>

#define ONESECOND 1000  // in ms
#define HALFSECOND 500  // in ms

enum CLOCK_MODE {
  CLOCK_MODE_CLOCK64,
  CLOCK_MODE_WALL_CLOCK64
};

__global__ void kernel_c(int clockRate, uint64_t wait_t) {
  uint64_t start = clock64() / clockRate, cur = 0; // in ms
  do { cur = clock64() / clockRate-start;} while (cur < wait_t);
}

__global__ void kernel_w(int clockRate, uint64_t wait_t) {
  uint64_t start = wall_clock64() / clockRate, cur = 0; // in ms
  do { cur = wall_clock64() / clockRate-start;} while (cur < wait_t);
}

bool verifyTimeExecution(CLOCK_MODE m, float time1, float time2,
                         float expectedTime1, float expectedTime2) {
  bool testStatus = false;
  float ratio = m == CLOCK_MODE_CLOCK64 ? 0.5 : 0.01;

  if (fabs(time1 - expectedTime1) < ratio * expectedTime1
      && fabs(time2 - expectedTime2) < ratio * expectedTime2) {
    WARN("Succeeded: Expected Vs Actual: Kernel1 - " << expectedTime1 << " Vs " << time1
         << ", Kernel2 - " << expectedTime2 << " Vs " << time2);
    testStatus = true;
  } else {
    FAIL_CHECK("Failed: Expected Vs Actual: Kernel1 -" << expectedTime1 << " Vs " << time1
         << ", Kernel2 - " << expectedTime2 << " Vs " << time2);
    testStatus = false;
  }
  return testStatus;
}

/*
 * Launching kernel1 and kernel2 and then we try to
 * get the event elapsed time of each kernel using the start and
 * end events.The event elapsed time should return us the kernel
 * execution time for that particular kernel
*/
bool kernelTimeExecution(CLOCK_MODE m, int clockRate,
                         uint64_t expectedTime1, uint64_t expectedTime2) {
  hipStream_t stream;
  hipEvent_t start_event1, end_event1, start_event2, end_event2;
  float time1 = 0, time2 = 0;
  HIPCHECK(hipEventCreate(&start_event1));
  HIPCHECK(hipEventCreate(&end_event1));
  HIPCHECK(hipEventCreate(&start_event2));
  HIPCHECK(hipEventCreate(&end_event2));
  HIPCHECK(hipStreamCreate(&stream));
  hipExtLaunchKernelGGL( m == CLOCK_MODE_CLOCK64 ? kernel_c : kernel_w,
      dim3(1), dim3(1), 0, stream, start_event1, end_event1, 0, clockRate, expectedTime1);
  hipExtLaunchKernelGGL( m == CLOCK_MODE_CLOCK64 ? kernel_c : kernel_w,
      dim3(1), dim3(1), 0, stream, start_event2, end_event2, 0, clockRate, expectedTime2);
  HIPCHECK(hipStreamSynchronize(stream));
  HIPCHECK(hipEventElapsedTime(&time1, start_event1, end_event1));
  HIPCHECK(hipEventElapsedTime(&time2, start_event2, end_event2));

  HIPCHECK(hipStreamDestroy(stream));
  HIPCHECK(hipEventDestroy(start_event1));
  HIPCHECK(hipEventDestroy(end_event1));
  HIPCHECK(hipEventDestroy(start_event2));
  HIPCHECK(hipEventDestroy(end_event2));

  return verifyTimeExecution(m, time1, time2, expectedTime1, expectedTime2);
}

TEST_CASE("Unit_hipClock64_Check")  {
  HIPCHECK(hipSetDevice(0));
  int clockRate = 0; // in KHz
  HIPCHECK(hipDeviceGetAttribute(&clockRate, hipDeviceAttributeClockRate, 0));

  SECTION("Verify kernel execution time via clock64()") {
    CHECK(kernelTimeExecution(CLOCK_MODE_CLOCK64, clockRate, ONESECOND, HALFSECOND));
  }
}

TEST_CASE("Unit_hipWallClock64_Check")  {
  HIPCHECK(hipSetDevice(0));
  int clockRate = 0; // in KHz
  hipDeviceGetAttribute(&clockRate, hipDeviceAttributeWallClockRate, 0);

  if(!clockRate) {
    INFO("hipDeviceAttributeWallClockRate has not been supported. Skipped");
    return;
  }

  SECTION("Verify kernel execution time via wall_clock64()") {
    CHECK(kernelTimeExecution(CLOCK_MODE_WALL_CLOCK64, clockRate, ONESECOND, HALFSECOND));
  }
}
