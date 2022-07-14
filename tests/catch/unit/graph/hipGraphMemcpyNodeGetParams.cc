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
1) Create a graph, add Memcpy node to graph with desired node params.
   Verify api fetches the node params mentioned while adding Memcpy node.
2) Set Memcpy node params with hipGraphMemcpyNodeSetParams,
   now get the params and verify both are same.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define SIZE 10
#define UPDATESIZE 8

/* Test verifies hipGraphMemcpyNodeGetParams API Negative scenarios.
 */
TEST_CASE("Unit_hipGraphMemcpyNodeGetParams_Negative") {
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
    ret = hipGraphMemcpyNodeGetParams(nullptr, &myparms);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass un-initilize node") {
    hipGraphNode_t memcpyNode_uninit{};
    ret = hipGraphMemcpyNodeGetParams(memcpyNode_uninit, &myparms);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass GetNodeParams as nullptr") {
    ret = hipGraphMemcpyNodeGetParams(memcpyNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipFreeArray(devArray));
  free(hData);
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipGraphMemcpyNodeGetParams API Functional scenarios.
 */

static bool compareHipPos(hipPos hPos1, hipPos hPos2) {
  if ((hPos1.x == hPos2.x) && (hPos1.y == hPos2.y) && (hPos1.z == hPos2.z))
    return true;
  else
    return false;
}
static bool compareHipExtent(hipExtent hExt1, hipExtent hExt2) {
  if ((hExt1.width == hExt2.width) && (hExt1.height == hExt2.height) &&
      (hExt1.depth == hExt2.depth))
    return true;
  else
    return false;
}
static bool compareHipPitchedPtr(hipPitchedPtr hpPtr1, hipPitchedPtr hpPtr2) {
  if ((reinterpret_cast<int *>(hpPtr1.ptr) ==
       reinterpret_cast<int *>(hpPtr2.ptr))
       && (hpPtr1.pitch == hpPtr2.pitch)
       #if HT_AMD
       && (hpPtr1.xsize == hpPtr2.xsize)
       /* xsize check below is disabled on nvidia as xsize value
        * is not being updated properly due to issue with CUDA api */
       #endif
       && (hpPtr1.ysize == hpPtr2.ysize))
    return true;
  else
    return false;
}

static bool memcpyNodeCompare(hipMemcpy3DParms *mNode1,
                              hipMemcpy3DParms *mNode2) {
  if (mNode1->srcArray != mNode2->srcArray)
    return false;
  if (!compareHipPos(mNode1->srcPos, mNode2->srcPos))
    return false;
  if (!compareHipPitchedPtr(mNode1->srcPtr, mNode2->srcPtr))
    return false;
  if (mNode1->dstArray != mNode2->dstArray)
    return false;
  if (!compareHipPos(mNode1->dstPos, mNode2->dstPos))
    return false;
  if (!compareHipPitchedPtr(mNode1->dstPtr, mNode2->dstPtr))
    return false;
  if (!compareHipExtent(mNode1->extent, mNode2->extent))
    return false;
  if (mNode1->kind != mNode2->kind)
    return false;
  return true;
}

TEST_CASE("Unit_hipGraphMemcpyNodeGetParams_Functional") {
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
  hipGraphNode_t memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &myparms));

  SECTION("Get Memcpy Param and verify.") {
    hipMemcpy3DParms m3DGetParams;
    REQUIRE(hipSuccess == hipGraphMemcpyNodeGetParams(memcpyNode,
                                                      &m3DGetParams));
    // Validating the result
    REQUIRE(true == memcpyNodeCompare(&myparms, &m3DGetParams));
  }

  SECTION("Set memcpy params and Get param and verify.") {
    hipMemcpy3DParms myparms1, m3DGetParams1;
    constexpr int width1{UPDATESIZE}, height1{UPDATESIZE}, depth1{UPDATESIZE};
    hipArray *devArray1;
    hipChannelFormatKind formatKind1 = hipChannelFormatKindSigned;
    int* hData1;
    uint32_t size1 = width1 * height1 * depth1 * sizeof(int);
    hData1 = reinterpret_cast<int*>(malloc(size1));
    REQUIRE(hData1 != nullptr);
    memset(hData1, 0, size1);
    for (int i = 0; i < depth1; i++) {
      for (int j = 0; j < height1; j++) {
        for (int k = 0; k < width1; k++) {
          hData1[i*width1*height1 + j*width1 + k] = i*width1*height1 +
                                                    j*width1 + k;
        }
      }
    }
    hipChannelFormatDesc channelDesc1 = hipCreateChannelDesc(sizeof(int)*8,
                                              0, 0, 0, formatKind1);
    HIP_CHECK(hipMalloc3DArray(&devArray1, &channelDesc1,
              make_hipExtent(width1, height1, depth1), hipArrayDefault));
    memset(&myparms1, 0x0, sizeof(hipMemcpy3DParms));
    myparms1.srcPos = make_hipPos(0, 0, 0);
    myparms1.dstPos = make_hipPos(0, 0, 0);
    myparms1.extent = make_hipExtent(width1 , height1, depth1);
    myparms1.srcPtr = make_hipPitchedPtr(hData1, width1 * sizeof(int),
                                         width1, height1);
    myparms1.dstArray = devArray1;
    myparms1.kind = hipMemcpyHostToDevice;

    REQUIRE(hipSuccess == hipGraphMemcpyNodeSetParams(memcpyNode, &myparms1));
    REQUIRE(hipSuccess == hipGraphMemcpyNodeGetParams(memcpyNode,
                                                      &m3DGetParams1));
    REQUIRE(true == memcpyNodeCompare(&myparms1, &m3DGetParams1));

    HIP_CHECK(hipFreeArray(devArray1));
    free(hData1);
  }
  HIP_CHECK(hipFreeArray(devArray));
  free(hData);
  HIP_CHECK(hipGraphDestroy(graph));
}
