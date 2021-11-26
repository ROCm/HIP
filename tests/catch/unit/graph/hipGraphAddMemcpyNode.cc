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
 1) Add multiple Memcpy nodes to graph and verify node execution is
 working as expected.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/**
 * Functional Test adds memcpy nodes of types H2D, D2D and D2H to graph
 * and verifies execution sequence by launching graph.
 */
TEST_CASE("Unit_hipGraphAddMemcpyNode_Functional") {
  constexpr int width{10}, height{10}, depth{10};
  hipArray *devArray1, *devArray2;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparams;
  uint32_t size = width * height * depth * sizeof(int);
  hipGraph_t graph;
  hipGraphNode_t memcpyNode;
  std::vector<hipGraphNode_t> dependencies;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;

  int *hData = reinterpret_cast<int*>(malloc(size));
  int *hOutputData = reinterpret_cast<int *>(malloc(size));

  REQUIRE(hData != nullptr);
  REQUIRE(hOutputData != nullptr);
  memset(hData, 0, size);
  memset(hOutputData, 0,  size);

  HIP_CHECK(hipStreamCreate(&streamForGraph));

  // Initialize host buffer
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        hData[i*width*height + j*width + k] = i*width*height + j*width + k;
      }
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int)*8,
                                                          0, 0, 0, formatKind);
  HIP_CHECK(hipMalloc3DArray(&devArray1, &channelDesc,
                       make_hipExtent(width, height, depth), hipArrayDefault));
  HIP_CHECK(hipMalloc3DArray(&devArray2, &channelDesc,
                       make_hipExtent(width, height, depth), hipArrayDefault));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Host to Device
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width , height, depth);
  myparams.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
  myparams.dstArray = devArray1;
  myparams.kind = hipMemcpyHostToDevice;


  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &myparams));
  dependencies.push_back(memcpyNode);

  // Device to Device
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.srcArray = devArray1;
  myparams.dstArray = devArray2;
  myparams.extent = make_hipExtent(width, height, depth);
  myparams.kind = hipMemcpyDeviceToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                             dependencies.size(), &myparams));
  dependencies.clear();
  dependencies.push_back(memcpyNode);

  // Device to host
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.dstPtr = make_hipPitchedPtr(hOutputData, width * sizeof(int),
                                      width, height);
  myparams.srcArray = devArray2;
  myparams.extent = make_hipExtent(width, height, depth);
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                             dependencies.size(), &myparams));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Check result
  HipTest::checkArray(hData, hOutputData, width, height, depth);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  hipFreeArray(devArray1);
  hipFreeArray(devArray2);
  free(hData);
  free(hOutputData);
}
