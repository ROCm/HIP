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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

TEST_CASE("Unit_hipStreamPerThread_EventRecord") {
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipEventRecord(event, hipStreamPerThread));
}

__global__ void update_even_odd(unsigned int N, int* out) {
  for (unsigned int i = 0; i < N; ++i) {
    if (i%2 == 0) {
      out[i] = 2;
    } else {
      out[i] = 3;
    }
  }
}
TEST_CASE("Unit_hipStreamPerThread_EventSynchronize") {
  int* A_h = nullptr;
  int* A_d = nullptr;
  unsigned int size = 1000;

  HIP_CHECK(hipHostMalloc(&A_h, size*sizeof(int)));
  HIP_CHECK(hipMalloc(&A_d, size * sizeof(int)));

  hipEvent_t start, end;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&end));

  HIP_CHECK(hipEventRecord(start, hipStreamPerThread));
  update_even_odd<<<1, 1>>>(size, A_d);
  HIP_CHECK(hipEventRecord(end, hipStreamPerThread));

  HIP_CHECK(hipEventSynchronize(end));
  HIP_CHECK(hipMemcpy(A_h, A_d, size*sizeof(int), hipMemcpyDeviceToHost));

  // Verify result
  for (unsigned int i = 0; i < size; ++i) {
    if (i%2 == 0 && A_h[i] != 2)
      REQUIRE(false);
    else if (i%2 != 0 && A_h[i] != 3) {
      REQUIRE(false);
    }
  }
}