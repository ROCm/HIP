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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
 1) Create and add empty node to graph and verify addition is successful.
*/

#include <hip_test_common.hh>

/**
 * Functional Test to add empty node with dependencies
 */
TEST_CASE("Unit_hipGraphAddEmptyNode_Functional") {
  char *pOutBuff_d{};
  constexpr size_t size = 1024;
  hipGraph_t graph{};
  hipGraphNode_t memsetNode{}, emptyNode{};
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipMalloc(&pOutBuff_d, size));
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(pOutBuff_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = size * sizeof(char);
  memsetParams.height = 1;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                                              &memsetParams));
  dependencies.push_back(memsetNode);

  // Create emptyNode and add it to graph with dependency
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, dependencies.data(),
                                                        dependencies.size()));

  REQUIRE(emptyNode != nullptr);
  HIP_CHECK(hipFree(pOutBuff_d));
  HIP_CHECK(hipGraphDestroy(graph));
}
