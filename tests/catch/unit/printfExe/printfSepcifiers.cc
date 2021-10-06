/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>

__global__ void test_kernel() {
  const char* N = nullptr;
  const char* s = "hello world";
  printf("xyzzy\n");
  printf("%%\n");
  printf("hello %% world\n");
  printf("%%s\n");
  // Two special tests to make sure that the compiler pass correctly
  // skips over a '%%' without affecting the logic for locating
  // string arguments.
  printf("%%s%p\n", (void*)0xf01dab1eca55e77e);
  printf("%%c%s\n", "xyzzy");
  printf("%c%c%c\n", 's', 'e', 'p');
  printf("%d\n", -42);
  printf("%u\n", 42);
  printf("%f\n", 123.456);
#ifdef __HIP_PLATFORM_AMD__
  printf("%F\n", -123.456);
#else
  printf("%f\n", -123.456);
#endif
  printf("%e\n", -123.456);
  printf("%E\n", 123.456);
  printf("%g\n", 123.456);
  printf("%G\n", -123.456);
  printf("%c\n", 'x');
  printf("%s\n", N);
  printf("%p\n", (void *)N);
#ifdef __HIP_PLATFORM_AMD__
  printf("%.*f %*.*s %p\n", 8, 3.14159, 8, 5, s, (void*)0xf01dab1eca55e77e);
#else
  // In Cuda, printf doesn't support %.*, %*.*
  printf("%.8f %8.5s %p\n", 3.14159, s, (void*)0xf01dab1eca55e77e);
#endif
}

int main() {
  test_kernel<<<1, 1>>>();
  hipDeviceSynchronize();
  return 0;
}
