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
Functional-
1) Instantiate a graph with memcpy node, obtain executable graph and update the hipMemcpy3DParms node params with set. Make sure they are taking effect.
Negative-
1) Pass hGraphExec as nullptr and verify api returns error code.
2) Pass node as nullptr and verify api returns error code.
3) Pass pNodeParams as nullptr and verify api returns error code.
4) Pass pNodeParams as empty structure object and verify api returns error code.
5) API expects atleast one memcpy src pointer to be set. When hipMemcpy3DParms::srcArray and hipMemcpy3DParms::srcPtr.ptr both are nullptr, api expected to return error code.
6) API expects atleast one memcpy dst pointer to be set. When hipMemcpy3DParms::dstArray and hipMemcpy3DParms::dstPtr.ptr both are nullptr, api expected to return error code.
7) Passing different element size for hipMemcpy3DParms::srcArray and hipMemcpy3DParms::dstArray is expected to return error code.
8) Pass node of different graph and verify api returns error code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/* Test verifies hipGraphExecMemcpyNodeSetParams API Negative scenarios.
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams_Negative") {
  constexpr int width{10}, height{10}, depth{10};
  hipArray *devArray, *devArray2;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparms;
  hipError_t ret;
  int* hData;
  uint32_t size = width * height * depth * sizeof(int);
  hData = reinterpret_cast<int*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        hData[i*width*height + j*width + k] = i*width*height + j*width + k;
      }
    }
  }
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int)*8,
                                              0, 0, 0, formatKind);
  HIP_CHECK(hipMalloc3DArray(&devArray, &channelDesc, make_hipExtent(width,
                             height, depth), hipArrayDefault));
  HIP_CHECK(hipMalloc3DArray(&devArray2, &channelDesc, make_hipExtent(width+1,
                             height+1, depth+1), hipArrayDefault));
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.extent = make_hipExtent(width , height, depth);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
  myparms.dstArray = devArray;
  myparms.kind = hipMemcpyHostToDevice;

  hipGraph_t graph;
  hipGraphNode_t memcpyNode;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &myparms));

  // Instantiate the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  SECTION("Pass hGraphExec as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParams(nullptr, memcpyNode, &myparms);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass node as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParams(graphExec, nullptr, &myparms);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pNodeParams as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pNodeParams as empty structure object") {
    hipMemcpy3DParms temp{};
    ret = hipGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &temp);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy src pointer to be set") {
    hipMemcpy3DParms temp;
    memset(&temp, 0x0, sizeof(hipMemcpy3DParms));
    temp.srcPos = make_hipPos(0, 0, 0);
    temp.dstPos = make_hipPos(0, 0, 0);
    temp.extent = make_hipExtent(width , height, depth);
    temp.dstArray = devArray;
    temp.kind = hipMemcpyHostToDevice;
    ret = hipGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &temp);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy dst pointer to be set") {
    hipMemcpy3DParms temp;
    memset(&temp, 0x0, sizeof(hipMemcpy3DParms));
    temp.srcPos = make_hipPos(0, 0, 0);
    temp.dstPos = make_hipPos(0, 0, 0);
    temp.extent = make_hipExtent(width , height, depth);
    temp.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
    temp.kind = hipMemcpyHostToDevice;
    ret = hipGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &temp);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Passing different element size for hipMemcpy3DParms::srcArray"
                   "and hipMemcpy3DParms::dstArray") {
    hipMemcpy3DParms temp;
    memset(&temp, 0x0, sizeof(hipMemcpy3DParms));
    temp.srcPos = make_hipPos(0, 0, 0);
    temp.dstPos = make_hipPos(0, 0, 0);
    temp.extent = make_hipExtent(width , height, depth);
    temp.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
    temp.kind = hipMemcpyHostToDevice;
    temp.srcArray = devArray;
    temp.dstArray = devArray2;
    ret = hipGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &temp);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Check with other graph node") {
    hipGraph_t graph1;
    hipGraphNode_t memcpyNode1;
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode1, graph1, NULL, 0, &myparms));
    ret = hipGraphExecMemcpyNodeSetParams(graphExec, memcpyNode1, &myparms);
    REQUIRE(hipErrorInvalidValue == ret);
    HIP_CHECK(hipGraphDestroy(graph1));
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFreeArray(devArray));
  HIP_CHECK(hipFreeArray(devArray2));
  free(hData);
}

/* Test verifies hipGraphExecMemcpyNodeSetParams API Functional scenarios.
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams_Functional") {
  constexpr int XSIZE = 1024;
  int harray1D[XSIZE]{};
  int harray1Dres[XSIZE]{};
  constexpr int width{XSIZE};
  hipArray *devArray1, *devArray2;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparams;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode;
  std::vector<hipGraphNode_t> dependencies;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Initialize 1D object
  for (int i = 0; i < XSIZE; i++) {
    harray1D[i] = i + 1;
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int)*8,
                                                          0, 0, 0, formatKind);
  // Allocate 1D device array by passing depth(0), height(0)
  HIP_CHECK(hipMalloc3DArray(&devArray1, &channelDesc,
                       make_hipExtent(width, 0, 0), hipArrayDefault));
  HIP_CHECK(hipMalloc3DArray(&devArray2, &channelDesc,
                       make_hipExtent(width, 0, 0), hipArrayDefault));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Host to Device
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.srcPtr = make_hipPitchedPtr(harray1D, width * sizeof(int),
                                       width, 1);
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
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.kind = hipMemcpyDeviceToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                  dependencies.size(), &myparams));
  dependencies.clear();
  dependencies.push_back(memcpyNode);

  // Device to host
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.dstPtr = make_hipPitchedPtr(harray1Dres, width * sizeof(int),
                                       width, 1);
  myparams.srcArray = devArray2;
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                  dependencies.size(), &myparams));

  // Instantiate the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  int harray1Dupdate[XSIZE]{};
  hipArray *devArray3;
  HIP_CHECK(hipMalloc3DArray(&devArray3, &channelDesc,
                       make_hipExtent(width, 0, 0), hipArrayDefault));

  // D2H updated with different pointer harray1Dres -> harray1Dupdate
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, 1, 1);
  myparams.dstPtr = make_hipPitchedPtr(harray1Dupdate, width * sizeof(int),
                                      width, 1);
  myparams.srcArray = devArray2;
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &myparams));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Validate result
  for (int i = 0; i < XSIZE; i++) {
    if (harray1D[i] != harray1Dupdate[i]) {
      INFO("harray1D: " << harray1D[i] << " harray1Dupdate: " <<
                      harray1Dupdate[i] << " mismatch at : " << i);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFreeArray(devArray1));
  HIP_CHECK(hipFreeArray(devArray2));
}
