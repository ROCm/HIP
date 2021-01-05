/*
   Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

bool hipEvent_Nullcheck() {
  bool TestStatus = true;
  hipError_t err;
  hipEvent_t start_event;
  // Passing nullptr to hipEventCreate API
  err = hipEventCreate(nullptr);
  if (err == hipSuccess) {
    printf("hipEventCreate failed when nullptr is passed \n");
    TestStatus = false;
  }
  // Passing nullptr to hipEventCreateWithFlags API
  err = hipEventCreateWithFlags(nullptr, 0);
  if (err == hipSuccess) {
    printf("hipEventCreatewithFlags failed when nullptr is passed \n");
    TestStatus = false;
  }
  // Passing illegal/unknown flag to hipEventCreateWithFlags API
  err = hipEventCreateWithFlags(&start_event, 10);
  if (err == hipSuccess) {
    printf("hipEventCreatewithFlags failed when illegal flag is passed \n");
    TestStatus = false;
  }
  // Passing nullptr to hipEventSynchronize API
  err = hipEventSynchronize(nullptr);
  if (err == hipSuccess) {
    printf("hipEventSynchronize failed when nullptr is passed \n");
    TestStatus = false;
  }
  // Passing nullptr to hipEventQuery API
  err = hipEventQuery(nullptr);
  if (err == hipSuccess) {
    printf("hipEventQuery failed when nullptr is passed \n");
    TestStatus = false;
  }
  // Passing nullptr to hipEventDestroy API
  err = hipEventDestroy(nullptr);
  if (err == hipSuccess) {
    printf("hipEventDestroy failed when nullptr is passed \n");
    TestStatus = false;
  }
  return TestStatus;
}

int main() {
  bool TestPassed = true;
  TestPassed = hipEvent_Nullcheck();
  if (TestPassed) {
    passed();
  } else {
    failed("Test Failed!");
  }
}

