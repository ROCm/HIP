/*
 * Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */

// Testcase Description: This test case tests if Host thread continues with
// next command after hipStreamAddCallback() api or wait for callback() call to
// finish. Ideally Host thread should not wait for callback to finish.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include <unistd.h>
#include <stdio.h>
#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_HCC__
#define HIPRT_CB
#endif

bool Callback_Completed = false;

void HIPRT_CB Callback1(hipStream_t stream, hipError_t status, void* userData) {
  sleep(5);
  Callback_Completed = true;
}

int main(int argc, char* argv[]) {
  hipStream_t mystream;
  HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));
  HIPCHECK(hipStreamAddCallback(mystream, Callback1, NULL, 0));
  sleep(1);

  // Callback_Completed is initialized to false.  The same is set to true at
  // the end of callback and callback sleeps for 5 seconds.
  // So, in case Callback_Completed is true here, it means the main thread
  // has waited till callback is complete and is a fail case.
  if (Callback_Completed == false) {
    HIPCHECK(hipStreamDestroy(mystream));
    passed();
  } else {
    HIPCHECK(hipStreamDestroy(mystream));
    failed("Unexpected: Host thread is waiting for callback to finish");
  }
}
