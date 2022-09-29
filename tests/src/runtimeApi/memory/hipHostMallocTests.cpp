/*
Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :

 (TestCase 1)::
 1) Test hipHostMalloc() api with ptr as nullptr and check for return value.
 2) Test hipHostMalloc() api with size as max(size_t) and check for OOM error.
 3) Test hipHostMalloc() api with flags as max(unsigned int) and validate
 return value.
 4) Pass size as zero for hipHostMalloc() api and check ptr is reset with
 with return value success.

*/

/* HIT_START
 * BUILD_CMD: %t %hc %S/%s %S/../../test_common.cpp -I%S/../../ -o %T/%t -std=c++11
 * TEST:
 * HIT_END
 */

#include "test_common.h"
#define NUM_BYTES 1000

int main(int argc, char *argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;
  hipError_t ret;
  size_t allocSize = NUM_BYTES;
  char *ptr;

  // Pass ptr as nullptr.
  if ((ret = hipHostMalloc(static_cast<void **>(nullptr), allocSize))
      != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropritate error value returned for "
           "ptr as nullptr. Error: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  // Size as max(size_t).
  if ((ret = hipHostMalloc(&ptr,
      std::numeric_limits<std::size_t>::max()))
      != hipErrorOutOfMemory) {
    printf("ArgValidation : Inappropritate error value returned for "
           "max(size_t). Error: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  // Flags as max(uint).
  if ((ret = hipHostMalloc(&ptr, allocSize,
      std::numeric_limits<unsigned int>::max()))
      != hipErrorInvalidValue) {
    printf("ArgValidation : Inappropritate error value returned for "
           "max(uint). Error: '%s'(%d)\n",
           hipGetErrorString(ret), ret);
    TestPassed &= false;
  }

  // Pass size as zero and check ptr reset.
  HIPCHECK(hipHostMalloc(&ptr, 0));
  if (ptr) {
    TestPassed &= false;
    printf("ArgValidation : ptr is not reset when size(0)\n");
  }

  if (TestPassed) {
    passed();
  } else {
    failed("hipHostMallocTests validation Failed!");
  }
}
