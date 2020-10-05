/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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


/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */


#include "test_common.h"

int main(int argc, char *argv[]) {
  int numDevices;
  hipGetDeviceCount(&numDevices);
  for (int i = 0; i < numDevices; i++) {
    hipStream_t stream;
    int priority;
    int priority_normal;
    int priority_low;
    int priority_high;
    int priority_check;

    // Test is to get the Stream Priority Range
    HIPCHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    printf("Priority range is %d for low and %d for high \n", priority_low, priority_high);
    priority_normal = priority_low + priority_high;

    // Check if priorities are indeed supported
    if ((priority_low + priority_high) != 0) {
      passed(); // exit the test since priorities are not supported
    }

    // Checking Priority of default stream
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamGetPriority(stream, &priority));
    if (priority_normal != priority) {
      failed("Unable to set Normal Priority for the stream");
    }
    HIPCHECK(hipStreamDestroy(stream));

    // Creating Stream with Priorities
    HIPCHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_high));
    HIPCHECK(hipStreamGetPriority(stream, &priority_check));
    if (priority_check != priority_high) {
      failed("Unable to set high priority for the stream");
    }
    HIPCHECK(hipStreamDestroy(stream));

    HIPCHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_low));
    HIPCHECK(hipStreamGetPriority(stream, &priority_check));
    if (priority_check != priority_low) {
      failed("Unable to set low priority for the stream");
    }
    HIPCHECK(hipStreamDestroy(stream));

    // creating a stream with boundry cases
    HIPCHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_low+1));
    HIPCHECK(hipStreamGetPriority(stream, &priority_check));
    if (priority_check != priority_low) {
      failed("setting priority failed ");
    }
    HIPCHECK(hipStreamDestroy(stream));

    HIPCHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_high-1));
    HIPCHECK(hipStreamGetPriority(stream, &priority_check));
    if (priority_check != priority_high) {
      failed("setting priority failed ");
    }
    HIPCHECK(hipStreamDestroy(stream));
  }

  passed();
  return 0;
}
