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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11  EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

// Checks the callback execution in the same order it was added
// Also, it checks if the number of callbacks executed are same as the number
// of callbacks added

#include <stdio.h>
#include <atomic>
#include "hip/hip_runtime.h"
#include "test_common.h"
#ifdef __HIP_PLATFORM_HCC__
#define HIPRT_CB
#endif

#define NUM_CALLS 10
hipStream_t mystream;
bool Callback_SequenceMismatch = false;
std::atomic<int> Cb_ordinal{0};

void HIPRT_CB Stream_Callback(hipStream_t stream, hipError_t status,
                              void* userData) {
  // Userdata has the order of the callback.  It should match with
  // the callback counter Cb_ordinal as the sequence of callback
  // should match the sequence of callback addition
  if (*(reinterpret_cast<int*>(userData)) == Cb_ordinal) {
    // Increment the Cb_ordinal to prepare for next sequence
    Cb_ordinal++;
  } else {
    Callback_SequenceMismatch = true;
  }

  delete reinterpret_cast<int*>(userData);
}

int main(int argc, char* argv[]) {
  int *ptr;
  HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));
  for (int i = 0; i< NUM_CALLS; i++) {
    ptr = new int;
    *ptr = i;
    // Pass the userdata with the order of the callback addition
    HIPCHECK(hipStreamAddCallback(mystream, Stream_Callback,
                                  reinterpret_cast<void*>(ptr), 0));
  }

  HIPCHECK(hipStreamSynchronize(mystream));
  HIPCHECK(hipStreamDestroy(mystream));

  if (!(Cb_ordinal == (NUM_CALLS))) {
    failed("All callbacks for stream did not get called!");
  }

  if (Callback_SequenceMismatch == false) {
    passed();
  } else {
    failed("hipStreamAddCallback() calls did not execute in sequence!");
  }
}
