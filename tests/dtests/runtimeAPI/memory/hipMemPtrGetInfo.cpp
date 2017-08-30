/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include"test_common.h"

struct {
  float a;
  int b;
  void *c;
} Struct ;

int main(){
  int *iPtr;
  float *fPtr;
  struct Struct *sPtr;
  size_t sSetSize = 1024, sGetSize;
  hipMalloc(&iPtr, sSetSize);
  hipMalloc(&fPtr, sSetSize);
  hipMalloc(&sPtr, sSetSize);
  hipMemPtrGetInfo(iPtr, &sGetSize);
  assert(sGetSize == sSetSize);
  hipMemPtrGetInfo(fPtr, &sGetSize);
  assert(sGetSize == sSetSize);
  hipMemPtrGetInfo(sPtr, &sGetSize);
  assert(sGetSize == sSetSize);
  passed();
}
