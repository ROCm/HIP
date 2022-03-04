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
Negative -
1) Pass pGraphNode as nullptr and check if api returns error.
2) When graph is un-initialized argument(skipping graph creation), api should return error code.
3) Passing pDependencies as nullptr, api should return success.
4) When numDependencies is max(size_t) and pDependencies is not valid ptr, api expected to return error code.
5) When pDependencies is nullptr, but numDependencies is non-zero, api expected to return error.
6) When destination ptr  is nullptr, api expected to return error code.
7) When source ptr is nullptr, api expected to return error code.
8) If count is more than allocated size for source and destination ptr, error code is returned.
9) If count is less than or equal to allocated size of source and destination ptr, api should return success.
*/

#include <hip_test_common.hh>

/**
 * Negative Test for API hipGraphAddMemcpyNode1D
 */
TEST_CASE("Unit_hipGraphAddMemcpyNode1D_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  int *A_d, *A_h;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode{};
  hipError_t ret;

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&A_h, Nbytes));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  SECTION("Pass pGraphNode as nullptr") {
    ret = hipGraphAddMemcpyNode1D(nullptr, graph,
            nullptr, 0, A_d, A_h, Nbytes, hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass graph as nullptr") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, nullptr,
            nullptr, 0, A_d, A_h, Nbytes, hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pDependencies as nullptr") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, graph,
            nullptr, 0, A_d, A_h, Nbytes, hipMemcpyHostToDevice);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass numDependencies is max and pDependencies is not valid ptr") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, graph,
            nullptr, INT_MAX, A_d, A_h, Nbytes, hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pDependencies as nullptr, but numDependencies is non-zero") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, graph,
            nullptr, 9, A_d, A_h, Nbytes, hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass destination ptr as nullptr") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, graph,
            nullptr, 0, nullptr, A_h, Nbytes, hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass source ptr as nullptr") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, graph,
            nullptr, 0, A_d, nullptr, Nbytes, hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass count as more than allocated size for source ptr") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, graph,
            nullptr, 0, A_d, A_h, Nbytes+10, hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass count as less than allocated size for destination ptr") {
    ret = hipGraphAddMemcpyNode1D(&memcpyNode, graph,
            nullptr, 0, A_d, A_h, Nbytes-10, hipMemcpyHostToDevice);
    REQUIRE(hipSuccess == ret);
  }
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(A_h));
  HIP_CHECK(hipGraphDestroy(graph));
}
