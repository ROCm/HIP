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
 * BUILD: %t %s EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */

#include "test_common.h"
#include "printf_common.h"

__global__ void test_kernel() {
  printf("%16d\n", 42);
  printf("%.8d\n", 42);
  printf("%16.5d\n", -42);
  printf("%.8x\n", 0x42);
  printf("%.8o\n", 042);
  printf("%16.8e\n", 12345.67891);
  printf("%16.8f\n", -12345.67891);
  printf("%16.8g\n", 12345.67891);
  printf("%8.4e\n", -12345.67891);
  printf("%8.4f\n", 12345.67891);
  printf("%8.4g\n", 12345.67891);
  printf("%4.2f\n", 12345.67891);
  printf("%.1f\n", 12345.67891);
  printf("%.5s\n", "helloxyz");
}

int main(int argc, char **argv) {
  std::string reference(R"here(              42
00000042
          -00042
00000042
00000042
  1.23456789e+04
 -12345.67891000
       12345.679
-1.2346e+04
12345.6789
1.235e+04
12345.68
12345.7
hello
)here");

  CaptureStream capture(stdout);

  capture.Begin();
  hipLaunchKernelGGL(test_kernel, dim3(1), dim3(1), 0, 0);
  hipStreamSynchronize(0);
  capture.End();

  std::string device_output = capture.getData();

  HIPASSERT(device_output == reference);
  passed();
}
