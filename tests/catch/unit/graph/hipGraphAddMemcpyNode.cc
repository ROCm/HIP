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
Testcase Scenarios : Negative
1) Pass pGraphNode as nullptr and check if api returns error.
2) When graph is un-initialized argument(skipping graph creation),
   api should return error code.
3) Passing pDependencies as nullptr, api should return success.
4) When numDependencies is max(size_t) and pDependencies is not valid ptr,
   api expected to return error code.
5) When pDependencies is nullptr, but numDependencies is non-zero,
   api expected to return error.
6) When pCopyParams is nullptr, api expected to return error code.
7) API expects atleast one memcpy src pointer to be set.
   When hipMemcpy3DParms::srcArray and hipMemcpy3DParms::srcPtr.ptr both
   are nullptr, api expected to return error code.
8) API expects atleast one memcpy dst pointer to be set.
   When hipMemcpy3DParms::dstArray and hipMemcpy3DParms::dstPtr.ptr both
   are nullptr, api expected to return error code.
9) Passing different element size for hipMemcpy3DParms::srcArray and
   hipMemcpy3DParms::dstArray is expected to return error code.

Testcase Scenarios : Functional
1) Add memcpy node to graph and verify memcpy operation is success for all
   memcpy kinds(H2D, D2H and D2D).
   Memcpy nodes are added and assigned to default device.
2) Perform memcpy operation for 1D, 2D and 3D arrays on default device and
   verify the results.
3) Add memcpy node to graph and verify memcpy operation is success for all
   memcpy kinds(H2D, D2H and D2D).
   Memcpy nodes are added and assigned to Peer device.
4) Perform memcpy operation for 1D, 2D and 3D arrays on Peer device and
   verify the results.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define ZSIZE 32
#define YSIZE 32
#define XSIZE 32

/* Test verifies hipGraphAddMemcpyNode API Negative scenarios.
 */

TEST_CASE("Unit_hipGraphAddMemcpyNode_Negative") {
  constexpr int width{10}, height{10}, depth{10};
  hipArray *devArray1;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparams;
  uint32_t size = width * height * depth * sizeof(int);
  hipGraph_t graph;
  hipGraphNode_t memcpyNode;
  hipStream_t streamForGraph;
  hipError_t ret;

  int *hData = reinterpret_cast<int*>(malloc(size));
  int *hOutputData = reinterpret_cast<int *>(malloc(size));

  REQUIRE(hData != nullptr);
  REQUIRE(hOutputData != nullptr);
  memset(hData, 0, size);
  memset(hOutputData, 0,  size);

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));

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

  // Host to Device
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width , height, depth);
  myparams.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
  myparams.dstArray = devArray1;
  myparams.kind = hipMemcpyHostToDevice;

  SECTION("Pass pGraphNode as nullptr") {
    ret = hipGraphAddMemcpyNode(nullptr, graph, nullptr, 0, &myparams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("When graph is nullptr") {
    ret = hipGraphAddMemcpyNode(&memcpyNode, nullptr,  nullptr, 0, &myparams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Passing pDependencies as nullptr") {
    ret = hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &myparams);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("When numDependencies is max and pDependencies is not valid ptr") {
    ret = hipGraphAddMemcpyNode(&memcpyNode, graph,
                                nullptr, INT_MAX, &myparams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("When pDependencies is nullptr, but numDependencies is non-zero") {
    ret = hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 11, &myparams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pCopyParams as nullptr") {
    ret = hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy src pointer to be set") {
    memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
    myparams.srcPos = make_hipPos(0, 0, 0);
    myparams.dstPos = make_hipPos(0, 0, 0);
    myparams.extent = make_hipExtent(width , height, depth);
    myparams.dstArray = devArray1;
    myparams.kind = hipMemcpyHostToDevice;
    ret = hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &myparams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy dst pointer to be set") {
    memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
    myparams.srcPos = make_hipPos(0, 0, 0);
    myparams.dstPos = make_hipPos(0, 0, 0);
    myparams.extent = make_hipExtent(width , height, depth);
    myparams.srcPtr = make_hipPitchedPtr(hData, width * sizeof(int),
                                      width, height);
    myparams.kind = hipMemcpyHostToDevice;
    ret = hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &myparams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Passing different element size for hipMemcpy3DParms::srcArray"
                   "and hipMemcpy3DParms::dstArray") {
    myparams.srcArray = devArray1;
    hipArray *devArray2;
    HIP_CHECK(hipMalloc3DArray(&devArray2, &channelDesc,
              make_hipExtent(width+1, height+1, depth+1), hipArrayDefault));
    myparams.dstArray = devArray2;
    ret = hipGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &myparams);
    REQUIRE(hipErrorInvalidValue == ret);
    HIP_CHECK(hipFreeArray(devArray2));
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFreeArray(devArray1));
  free(hData);
  free(hOutputData);
}

static void validateMemcpyNode3DArray(bool peerAccess = false) {
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

  HIP_CHECK(hipSetDevice(0));
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

  // For peer access test, Memory is allocated on device(0)
  // while memcpy nodes are allocated and assigned to peer device(1)
  if (peerAccess) {
    HIP_CHECK(hipSetDevice(1));
  }

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
  HIP_CHECK(hipFreeArray(devArray1));
  HIP_CHECK(hipFreeArray(devArray2));
  free(hData);
  free(hOutputData);
}

static void validateMemcpyNode2DArray(bool peerAccess = false) {
  int harray2D[YSIZE][XSIZE]{};
  int harray2Dres[YSIZE][XSIZE]{};
  constexpr int width{XSIZE}, height{YSIZE};
  hipArray *devArray1, *devArray2;
  hipChannelFormatKind formatKind = hipChannelFormatKindSigned;
  hipMemcpy3DParms myparams;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode;
  std::vector<hipGraphNode_t> dependencies;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Initialize 2D object
  for (int i = 0; i < YSIZE; i++) {
    for (int j = 0; j < XSIZE; j++) {
      harray2D[i][j] = i + j + 1;
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int)*8,
                                                          0, 0, 0, formatKind);
  // Allocate 2D device array by passing depth(0)
  HIP_CHECK(hipMalloc3DArray(&devArray1, &channelDesc,
                       make_hipExtent(width, height, 0), hipArrayDefault));
  HIP_CHECK(hipMalloc3DArray(&devArray2, &channelDesc,
                       make_hipExtent(width, height, 0), hipArrayDefault));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // For peer access test, Memory is allocated on device(0)
  // while memcpy nodes are allocated and assigned to peer device(1)
  if (peerAccess) {
    HIP_CHECK(hipSetDevice(1));
  }

  // Host to Device
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, height, 1);
  myparams.srcPtr = make_hipPitchedPtr(harray2D, width * sizeof(int),
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
  myparams.extent = make_hipExtent(width, height, 1);
  myparams.kind = hipMemcpyDeviceToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                             dependencies.size(), &myparams));
  dependencies.clear();
  dependencies.push_back(memcpyNode);

  // Device to host
  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(width, height, 1);
  myparams.dstPtr = make_hipPitchedPtr(harray2Dres, width * sizeof(int),
                                      width, height);
  myparams.srcArray = devArray2;
  myparams.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, dependencies.data(),
                                             dependencies.size(), &myparams));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Validate result
  for (int i = 0; i < YSIZE; i++) {
    for (int j = 0; j < XSIZE; j++) {
      if (harray2D[i][j] != harray2Dres[i][j]) {
        INFO("harray2D: " << harray2D[i][j] << "harray2Dres: "
              << harray2Dres[i][j] << " mismatch at (i,j) : " << i << j);
        REQUIRE(false);
      }
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFreeArray(devArray1));
  HIP_CHECK(hipFreeArray(devArray2));
}

static void validateMemcpyNode1DArray(bool peerAccess = false) {
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

  HIP_CHECK(hipSetDevice(0));
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

  // For peer access test, Memory is allocated on device(0)
  // while memcpy nodes are allocated and assigned to peer device(1)
  if (peerAccess) {
    HIP_CHECK(hipSetDevice(1));
  }

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

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Validate result
  for (int i = 0; i < XSIZE; i++) {
    if (harray1D[i] != harray1Dres[i]) {
      INFO("harray1D: " << harray1D[i] << " harray1Dres: " << harray1Dres[i]
            << " mismatch at : " << i);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFreeArray(devArray1));
  HIP_CHECK(hipFreeArray(devArray2));
}

/**
 * Basic Functional Tests adds memcpy nodes of types H2D, D2D and D2H to graph
 * and verifies execution sequence by launching graph on default device.
 * Tests also verify memcpy node addition with 1D, 2D and 3D objects.
 */
TEST_CASE("Unit_hipGraphAddMemcpyNode_BasicFunctional") {
  SECTION("Memcpy with 3D array on default device") {
    validateMemcpyNode3DArray();
  }

  SECTION("Memcpy with 2D array on default device") {
    validateMemcpyNode2DArray();
  }

  SECTION("Memcpy with 1D array on default device") {
    validateMemcpyNode1DArray();
  }
}

/**
 * Peer access tests adds and assigns memcpy nodes of types H2D, D2D and D2H
 * to peer device. Memory allocations happen on device(0) and memcpy operations
 * are performed from device(1).
 * Tests also verify memcpy node addition with 1D, 2D and 3D objects.
 */
TEST_CASE("Unit_hipGraphAddMemcpyNode_PeerAccessFunctional") {
  int numDevices{}, peerAccess{};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
  }

  if (!peerAccess) {
    WARN("Skipping test as peer device access is not found!");
    return;
  }

  SECTION("Memcpy with 3D array on peer device") {
    validateMemcpyNode3DArray(true);
  }

  SECTION("Memcpy with 2D array on peer device") {
    validateMemcpyNode2DArray(true);
  }

  SECTION("Memcpy with 1D array on peer device") {
    validateMemcpyNode1DArray(true);
  }
}

