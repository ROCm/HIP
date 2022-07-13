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
Testcase Scenarios : hipLaunchBounds_With_maxThreadsPerBlock
1) Passing threadsPerBlock same as kernel launch_bounds.
2) Passing threadsPerBlock less than kernel launch_bounds.
3) Passing threadsPerBlock more than kernel launch_bounds.
4) Passing threadsPerBlock as 0 to kernel launch_bounds.
Testcase Scenarios : hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU
1) Passing threadsPerBlock same as kernel launch_bounds.
2) Passing threadsPerBlock less than kernel launch_bounds.
3) Passing threadsPerBlock more than kernel launch_bounds.
4) Passing threadsPerBlock as 0 to kernel launch_bounds.
5) Passing blocksPerCU same as kernel launch_bounds.
6) Passing blocksPerCU less than kernel launch_bounds.
7) Passing blocksPerCU more than kernel launch_bounds.
8) Passing blocksPerCU as 0 to kernel launch_bounds.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

__global__ void
__launch_bounds__(128, 2)
MyKernel(int N, int *x, int val) {
  for (int i = 0; i < N; i++) {
    x[i] = val;
  }
}

__global__ void
__launch_bounds__(64)
MyKernel_2(int N, int *x, int val) {
  for (int i = 0; i < N; i++) {
    x[i] = val;
  }
}

static bool verify(int N, int *x, int val) {
  for (int i = 0; i < N; i++) {
    if (x[i] != val) {
      return false;
    }
  }
  return true;
}

TEST_CASE("Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") {
  constexpr size_t N = 10000;
  hipError_t ret;
  int *x;

  HIP_CHECK(hipMallocManaged(&x, N*sizeof(int)));
  REQUIRE(x != nullptr);

  SECTION("Passing threadsPerBlock same as kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel_2, dim3(4), dim3(64), 0, 0, N, x, 2);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(N, x, 2));
  }
  SECTION("Passing threadsPerBlock less than kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel_2, dim3(4), dim3(32), 0, 0, N, x, 22);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(N, x, 22));
  }
  SECTION("Passing threadsPerBlock more than kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel_2, dim3(4), dim3(128), 0, 0, N, x, 9);
    ret = hipGetLastError();
    REQUIRE(hipSuccess != ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true != verify(N, x, 9));
  }
  SECTION("Passing threadsPerBlock as 0 to kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel_2, dim3(4), dim3(0), 0, 0, N, x, 19);
    ret = hipGetLastError();
    REQUIRE(hipSuccess != ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true != verify(N, x, 19));
  }

  HIP_CHECK(hipFree(x));
}

TEST_CASE("Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") {
  constexpr size_t N = 10000;
  hipError_t ret;
  int *x;

  HIP_CHECK(hipMallocManaged(&x, N*sizeof(int)));
  REQUIRE(x != nullptr);

  SECTION("Passing threadsPerBlock same as kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(1), dim3(128), 0, 0, N, x, 1);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(N, x, 1));
  }
  SECTION("Passing threadsPerBlock less than kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(2), dim3(64), 0, 0, N, x, 11);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(N, x, 11));
  }
  SECTION("Passing threadsPerBlock more than kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(2), dim3(256), 0, 0, N, x, 3);
    ret = hipGetLastError();
    REQUIRE(hipSuccess != ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true != verify(N, x, 3));
  }
  SECTION("Passing threadsPerBlock as 0 to kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(2), dim3(0), 0, 0, N, x, 13);
    ret = hipGetLastError();
    REQUIRE(hipSuccess != ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true != verify(N, x, 13));
  }

  SECTION("Passing blocksPerCU same as kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(2), dim3(128), 0, 0, N, x, 5);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(N, x, 5));
  }
  SECTION("Passing blocksPerCU less than kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(1), dim3(128), 0, 0, N, x, 25);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(N, x, 25));
  }
  SECTION("Passing blocksPerCU more than kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(4), dim3(128), 0, 0, N, x, 7);
    ret = hipGetLastError();
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == verify(N, x, 7));
  }
  SECTION("Passing blocksPerCU as 0 to kernel launch_bounds") {
    hipLaunchKernelGGL(MyKernel, dim3(0), dim3(128), 0, 0, N, x, 37);
    ret = hipGetLastError();
    REQUIRE(hipSuccess != ret);
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true != verify(N, x, 37));
  }

  HIP_CHECK(hipFree(x));
}

