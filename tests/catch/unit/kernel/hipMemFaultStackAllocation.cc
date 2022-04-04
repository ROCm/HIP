/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/**
Testcase Scenarios :
1) Calling Kernel which allocate ConstSize to local array.
2) Calling Kernel which allocate VariableSize to local array.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

const size_t N = 100000;

__global__ void MyKernelConstSize(int* C_d, const int* A_d) {
  constexpr size_t A1size = 1024;
  int A1[A1size];

  for (size_t i = 0; i < A1size; ++i) {
    A1[i] = i;
  }

  for (size_t i = 0; i < N; ++i) {
    C_d[i] = A_d[i] + A1[i%A1size];
  }
}

__global__ void MyKernelVariableSize(int* C_d, const int* A_d) {
  constexpr size_t A1size = 1024;
  int A1[1024];

  for (size_t i = 0; i < A1size; ++i) {
    A1[i] = i;
  }

  for (size_t i = 0; i < N; ++i) {
    C_d[i] = A_d[i] + A1[i%A1size];
  }
}

static bool verify(const int* C_d, const int* A_d) {
  for (size_t i = 0; i < N; i++) {
    if (C_d[i] != A_d[i] + i%1024) {
      return false;
    }
  }
  return true;
}

TEST_CASE("Unit_hipMemFaultStackAllocation_Check") {
  hipError_t ret;
  int *A_d, *C_d;
  const size_t Nbytes = N * sizeof(int);
  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (N + threadsPerBlock - 1)/threadsPerBlock;

  HIP_CHECK(hipMallocManaged(&A_d, Nbytes));
  REQUIRE(A_d != nullptr);
  HIP_CHECK(hipMallocManaged(&C_d, Nbytes));
  REQUIRE(C_d != nullptr);

  for (size_t i = 0; i < N; i++) {
    A_d[i] = i%1024;
  }

  SECTION("Calling Kernel which allocate ConstSize to local array") {
    hipLaunchKernelGGL(MyKernelConstSize, dim3(blocks),
                       dim3(threadsPerBlock), 0, 0, C_d, A_d);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(C_d, A_d));
  }
  SECTION("Calling Kernel which allocate VariableSize to local array") {
    hipLaunchKernelGGL(MyKernelVariableSize, dim3(blocks),
                       dim3(threadsPerBlock), 0, 0, C_d, A_d);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(C_d, A_d));
  }

  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(A_d));
}

