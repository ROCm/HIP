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
Negative -
1) Pass node as nullptr and verify api returns error code.
2) Pass un-initialize node and verify api returns error code.
3) Pass pNodeParams as nullptr and verify api returns error code.
Functional -
1) Add Memcpy node to graph, update the Memcpy node params with set and
   launch the graph and check updated params are taking effect.
2) Add Memcpy node to graph, launch graph, then update the Memcpy node params
   with set and launch the graph and check updated params are taking effect.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define SIZE 10

/* Test verifies hipGraphMemcpyNodeSetParams API Negative scenarios.
 */
TEST_CASE("Unit_hipGraphMemcpyNodeSetParams_Negative") {
  constexpr int width{SIZE}, height{SIZE}, depth{SIZE};
  hipArray *devArray;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparms;
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
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.extent = make_hipExtent(width , height, depth);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
  myparms.dstArray = devArray;
  myparms.kind = hipMemcpyHostToDevice;

  hipGraph_t graph;
  hipError_t ret;
  hipGraphNode_t memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &myparms));

  SECTION("Pass node as nullptr") {
    ret = hipGraphMemcpyNodeSetParams(nullptr, &myparms);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass un-initialize node") {
    hipGraphNode_t memcpyNode_uninit{};
    ret = hipGraphMemcpyNodeSetParams(memcpyNode_uninit, &myparms);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass SetNodeParams as nullptr") {
    ret = hipGraphMemcpyNodeSetParams(memcpyNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipFreeArray(devArray));
  free(hData);
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipGraphMemcpyNodeSetParams API Functional scenarios.
 */
TEST_CASE("Unit_hipGraphMemcpyNodeSetParams_Functional") {
  constexpr int width{SIZE}, height{SIZE}, depth{SIZE};
  hipArray *devArray;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparms, myparms1;
  uint32_t size = width * height * depth * sizeof(int);

  int *hData = reinterpret_cast<int*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);
  int *hDataTemp = reinterpret_cast<int*>(malloc(size));
  REQUIRE(hDataTemp != nullptr);
  memset(hDataTemp, 0, size);
  int *hOutputData = reinterpret_cast<int *>(malloc(size));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, 0,  size);
  int *hOutputData1 = reinterpret_cast<int *>(malloc(size));
  REQUIRE(hOutputData1 != nullptr);
  memset(hOutputData1, 0,  size);

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
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));

  // Host to Device
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.extent = make_hipExtent(width , height, depth);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
  myparms.dstArray = devArray;
  myparms.kind = hipMemcpyHostToDevice;

  hipGraph_t graph;
  hipGraphNode_t memcpyNode;
  std::vector<hipGraphNode_t> dependencies;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &myparms));
  dependencies.push_back(memcpyNode);

  // Device to host
  memset(&myparms1, 0x0, sizeof(hipMemcpy3DParms));
  myparms1.srcPos = make_hipPos(0, 0, 0);
  myparms1.dstPos = make_hipPos(0, 0, 0);
  myparms1.dstPtr = make_hipPitchedPtr(hDataTemp, width * sizeof(int),
                                      width, height);
  myparms1.srcArray = devArray;
  myparms1.extent = make_hipExtent(width, height, depth);
  myparms1.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                  dependencies.size(), &myparms1));

  SECTION("Update the memcpyNode and check") {
    // Device to host with updated host ptr hDataTemp -> hOutputData
    memset(&myparms1, 0x0, sizeof(hipMemcpy3DParms));
    myparms1.srcPos = make_hipPos(0, 0, 0);
    myparms1.dstPos = make_hipPos(0, 0, 0);
    myparms1.dstPtr = make_hipPitchedPtr(hOutputData, width * sizeof(int),
                                        width, height);
    myparms1.srcArray = devArray;
    myparms1.extent = make_hipExtent(width, height, depth);
    myparms1.kind = hipMemcpyDeviceToHost;

    HIP_CHECK(hipGraphMemcpyNodeSetParams(memcpyNode, &myparms1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));

    // Check result
    HipTest::checkArray(hData, hOutputData, width, height, depth);
  }

  SECTION("Update the memcpyNode again and check") {
    // Device to host with updated host ptr hOutputData -> hOutputData1
    memset(&myparms1, 0x0, sizeof(hipMemcpy3DParms));
    myparms1.srcPos = make_hipPos(0, 0, 0);
    myparms1.dstPos = make_hipPos(0, 0, 0);
    myparms1.dstPtr = make_hipPitchedPtr(hOutputData1, width * sizeof(int),
                                        width, height);
    myparms1.srcArray = devArray;
    myparms1.extent = make_hipExtent(width, height, depth);
    myparms1.kind = hipMemcpyDeviceToHost;

    HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                    dependencies.size(), &myparms1));
    HIP_CHECK(hipGraphMemcpyNodeSetParams(memcpyNode, &myparms1));

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));

    // Check result
    HipTest::checkArray(hData, hOutputData1, width, height, depth);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFreeArray(devArray));
  free(hData);
  free(hDataTemp);
  free(hOutputData);
  free(hOutputData1);
}
