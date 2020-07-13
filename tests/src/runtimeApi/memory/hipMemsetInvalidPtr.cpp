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
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
*/

//  * To test invalid pointer to hipMemset* apis

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#define N 50
#define MEMSETVAL 0x42
#define NUM_H 256
#define NUM_W 256

int main() {
  size_t Nbytes = N*sizeof(char);
  size_t pitch_A;
  size_t width = NUM_W * sizeof(char);
  size_t sizeElements = width * NUM_H;
  size_t elements = NUM_W * NUM_H;
  char *A_d;

  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width , NUM_H));

  hipError_t e;

  e = hipMemset(NULL , MEMSETVAL , Nbytes);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemsetD32(NULL , MEMSETVAL , Nbytes);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemsetD16(NULL , MEMSETVAL , Nbytes);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemsetD8(NULL , MEMSETVAL , Nbytes);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemsetAsync(NULL , MEMSETVAL , Nbytes , 0);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemsetD32Async(NULL , MEMSETVAL , Nbytes, 0);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemsetD16Async(NULL , MEMSETVAL , Nbytes, 0);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemsetD8Async(NULL , MEMSETVAL , Nbytes, 0);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemset2D(NULL, pitch_A, MEMSETVAL, NUM_W, NUM_H);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemset2DAsync(NULL, pitch_A, MEMSETVAL, NUM_W, NUM_H, 0);
  HIPASSERT(e == hipErrorInvalidValue);

  /* Passing host pointer to hipMemset.Ticket SWDEV-243206 is open for this.
   * Disabling this test until the ticket is closed
   *
  char *A_h;
  A_h = (char*)malloc(Nbytes);
  e = hipMemset(A_h, MEMSETVAL , Nbytes);
  HIPASSERT(e == hipErrorInvalidValue);
  */

  /* Passing invalid pitch to hipMemset2D.Ticket SWDEV-243104 is open for this.
   * Disabling this test until the ticket is closed
   *
  e = hipMemset2D(A_d, 0, MEMSETVAL, NUM_W, NUM_H);
  HIPASSERT(e == hipErrorInvalidValue);

  e = hipMemset2DAsync(A_d, 0, MEMSETVAL, NUM_W, NUM_H,0);
  HIPASSERT(e == hipErrorInvalidValue);
  */

  hipFree(A_d);
  passed();
}
