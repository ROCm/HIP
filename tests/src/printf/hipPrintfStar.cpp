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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s EXCLUDE_HIP_PLATFORM nvcc EXCLUDE_HIP_RUNTIME HCC EXCLUDE_HIP_COMPILER hcc
 * TEST: %t EXCLUDE_HIP_PLATFORM nvcc EXCLUDE_HIP_RUNTIME HCC EXCLUDE_HIP_COMPILER hcc
 * HIT_END
 */

#include "test_common.h"
#include "printf_common.h"

__global__ void test_kernel() {
  printf("%*d\n", 16, 42);
  printf("%.*d\n", 8, 42);
  printf("%*.*d\n", -16, 8, 42);
  printf("%*.*f %s * %.*s\n", 16, 8, 123.456, "hello", 5, "worldxyz");
}

int main(int argc, char **argv) {
  std::string reference(R"here(              42
00000042
00000042        
    123.45600000 hello * world
)here");

  CaptureStream captured(stdout);
  hipLaunchKernelGGL(test_kernel, dim3(1), dim3(1), 0, 0);
  hipStreamSynchronize(0);
  auto CapturedData = captured.getCapturedData();
  std::string device_output = gulp(CapturedData);

  HIPASSERT(device_output == reference);
  passed();
}
