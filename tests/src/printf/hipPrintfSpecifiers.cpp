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
  const char *N = nullptr;
  const char *s = "hello world";

  printf("xyzzy\n");
  printf("%%\n");
  printf("hello %% world\n");
  printf("%%s\n");
  // Two special tests to make sure that the compiler pass correctly
  // skips over a '%%' without affecting the logic for locating
  // string arguments.
  printf("%%s%p\n", (void *)0xf01dab1eca55e77e);
  printf("%%c%s\n", "xyzzy");
  printf("%c%c%c\n", 's', 'e', 'p');
  printf("%d\n", -42);
  printf("%u\n", 42);
  printf("%f\n", 123.456);
  printf("%F\n", -123.456);
  printf("%e\n", -123.456);
  printf("%E\n", 123.456);
  printf("%g\n", 123.456);
  printf("%G\n", -123.456);
  printf("%c\n", 'x');
  printf("%s\n", N);
  printf("%p\n", N);
  printf("%.*f %*.*s %p\n", 8, 3.14159, 8, 5, s, (void *)0xf01dab1eca55e77e);
}

int main(int argc, char **argv) {
#if !defined(_WIN32)
  std::string reference(R"here(xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
x

(nil)
3.14159000    hello 0xf01dab1eca55e77e
)here");
#else
  std::string reference(R"here(xyzzy
%
hello % world
%s
%sF01DAB1ECA55E77E
%cxyzzy
sep
-42
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
x

0000000000000000
3.14159000    hello F01DAB1ECA55E77E
)here");
#endif

  CaptureStream capture(stdout);

  capture.Begin();
  hipLaunchKernelGGL(test_kernel, dim3(1), dim3(1), 0, 0);
  hipStreamSynchronize(0);
  capture.End();

  std::string device_output = capture.getData();

  HIPASSERT(device_output == reference);
  passed();
}
