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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


/**
 * Functional Test for API - hipUserObjectCreate
1) Call hipUserObjectCreate once and release it by calling hipUserObjectRelease
2) Call hipUserObjectCreate refCount as X and release it by calling
   hipUserObjectRelease with same refCount.
3) Call hipUserObjectCreate, retain it by calling hipUserObjectRetain
   and release it by calling hipUserObjectRelease twice.
4) Call hipUserObjectCreate with refCount as X, retain it by calling
   hipUserObjectRetain with count as Y and release it by calling
   hipUserObjectRelease with count as X+Y.
 */

struct BoxStruct {
  int count;
  BoxStruct() {
    INFO("Constructor called for Struct!\n");
  }
  ~BoxStruct() {
    INFO("Destructor called for Struct!\n");
  }
};

class BoxClass {
 public:
  BoxClass() {
    INFO("Constructor called for Class!\n");
  }
  ~BoxClass() {
    INFO("Destructor called for Class!\n");
  }
};

static void destroyStructObj(void *ptr) {
  BoxStruct *ptr1 = reinterpret_cast<BoxStruct *>(ptr);
  delete ptr1;
}

static void destroyClassObj(void *ptr) {
  BoxClass *ptr2 = reinterpret_cast<BoxClass *>(ptr);
  delete ptr2;
}

static void destroyIntObj(void *ptr) {
  int *ptr2 = reinterpret_cast<int *>(ptr);
  delete ptr2;
}

static void destroyFloatObj(void *ptr) {
  float *ptr2 = reinterpret_cast<float *>(ptr);
  delete ptr2;
}

/* 1) Call hipUserObjectCreate once and release it by
      calling hipUserObjectRelease */
static void hipUserObjectCreate_Functional_1(void *object,
                                             void destroyObj(void *)) {
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyObj, 1,
                                  hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRelease(hObject));
}

TEST_CASE("Unit_hipUserObjectCreate_Functional_1") {
  SECTION("Called with int Object") {
    int *object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float *object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass *object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct *object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyStructObj);
  }
}

/* 2) Call hipUserObjectCreate refCount as X and release it by
      calling hipUserObjectRelease with same refCount. */
static void hipUserObjectCreate_Functional_2(void *object,
                                             void destroyObj(void *)) {
  int refCount = 5;
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyObj,
                                  refCount, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRelease(hObject, refCount));
}

TEST_CASE("Unit_hipUserObjectCreate_Functional_2") {
  SECTION("Called with int Object") {
    int *object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float *object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass *object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct *object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyStructObj);
  }
}

/* 3) Call hipUserObjectCreate, retain it by calling hipUserObjectRetain
      and release it by calling hipUserObjectRelease twice. */
static void hipUserObjectCreate_Functional_3(void *object,
                                             void destroyObj(void *)) {
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyObj,
                                  1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
}

TEST_CASE("Unit_hipUserObjectCreate_Functional_3") {
  SECTION("Called with int Object") {
    int *object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float *object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass *object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct *object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyStructObj);
  }
}

/* 4) Call hipUserObjectCreate with refCount as X, retain it by calling
      hipUserObjectRetain with count as Y and release it by calling
      hipUserObjectRelease with count as X+Y. */
static void hipUserObjectCreate_Functional_4(void *object,
                                             void destroyObj(void *)) {
  int refCount = 5;
  int refCountRetain = 8;
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyObj,
                                  refCount, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject, refCountRetain));
  HIP_CHECK(hipUserObjectRelease(hObject, refCount+refCountRetain));
}

TEST_CASE("Unit_hipUserObjectCreate_Functional_4") {
  SECTION("Called with int Object") {
    int *object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float *object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass *object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct *object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyStructObj);
  }
}


/**
 * Negative Test for API - hipUserObjectCreate
 1) Pass User Object as nullptr
 2) Pass object as nullptr
 3) Pass Callback function as nullptr
 4) Pass initialRefcount as 0
 5) Pass initialRefcount as INT_MAX
 6) Pass flag other than hipUserObjectNoDestructorSync
 */

TEST_CASE("Unit_hipUserObjectCreate_Negative") {
  hipError_t ret;
  int *object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  SECTION("Pass User Object as nullptr") {
    ret = hipUserObjectCreate(nullptr, object, destroyIntObj,
                               1, hipUserObjectNoDestructorSync);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass object as nullptr") {
    ret = hipUserObjectCreate(&hObject, nullptr, destroyIntObj,
                               1, hipUserObjectNoDestructorSync);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass Callback function as nullptr") {
    ret = hipUserObjectCreate(&hObject, object, nullptr,
                               1, hipUserObjectNoDestructorSync);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as 0") {
    ret = hipUserObjectCreate(&hObject, object, destroyIntObj,
                               0, hipUserObjectNoDestructorSync);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    ret = hipUserObjectCreate(&hObject, object, destroyIntObj,
                               INT_MAX, hipUserObjectNoDestructorSync);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass flag other than hipUserObjectNoDestructorSync") {
    ret = hipUserObjectCreate(&hObject, object, destroyIntObj,
                               1, hipUserObjectFlags(9));
    REQUIRE(hipErrorInvalidValue == ret);
  }
}

/**
 * Negative Test for API - hipUserObjectRelease
 1) Pass User Object as nullptr
 2) Pass initialRefcount as 0
 3) Pass initialRefcount as INT_MAX
 */

TEST_CASE("Unit_hipUserObjectRelease_Negative") {
  hipError_t ret;
  int *object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyIntObj,
                                  1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("Pass User Object as nullptr") {
    ret = hipUserObjectRelease(nullptr, 1);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as 0") {
    ret = hipUserObjectRelease(hObject, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    ret = hipUserObjectRelease(hObject, INT_MAX);
    REQUIRE(hipSuccess == ret);
  }
}

/**
 * Negative Test for API - hipUserObjectRetain
 1) Pass User Object as nullptr
 2) Pass initialRefcount as 0
 3) Pass initialRefcount as INT_MAX
 */

TEST_CASE("Unit_hipUserObjectRetain_Negative") {
  hipError_t ret;
  int *object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyIntObj,
                                  1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("Pass User Object as nullptr") {
    ret = hipUserObjectRetain(nullptr, 1);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as 0") {
    ret = hipUserObjectRetain(hObject, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    ret = hipUserObjectRetain(hObject, INT_MAX);
    REQUIRE(hipSuccess == ret);
  }
}

TEST_CASE("Unit_hipUserObj_Negative_Test") {
  hipError_t ret;
  int *object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  // Create a new hObject with 2 reference
  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyIntObj,
                                  2, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  // Release more than created.
  ret = hipUserObjectRelease(hObject, 4);
  REQUIRE(hipSuccess == ret);

  // Retain reference to a removed user object
  ret = hipUserObjectRetain(hObject, 1);
  REQUIRE(hipSuccess == ret);
}

/**
 * Functional Test for API - hipGraphRetainUserObject
 */

/* 1) Create GraphUserObject and retain it by calling hipGraphRetainUserObject
      and release it by calling hipGraphReleaseUserObject. */

static void hipGraphRetainUserObject_Functional_1(void *object,
                                                  void destroyObj(void *)) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyObj,
                                  1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 1,
                                       hipGraphUserObjectMove));

  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphRetainUserObject_Functional_1") {
  SECTION("Called with int Object") {
    int *object = new int();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float *object = new float();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass *object = new BoxClass();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct *object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipGraphRetainUserObject_Functional_1(object, destroyStructObj);
  }
}

/* 2) Create UserObject and GraphUserObject and retain using custom reference
      count and release it by calling hipGraphReleaseUserObject with count. */

TEST_CASE("Unit_hipGraphRetainUserObject_Functional_2") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode;
  hipKernelNodeParams kNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, dependencies.data(),
                                  dependencies.size(), &kNodeParams));

  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(),
                                    dependencies.size(), C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  int refCount = 2;
  int refCountRetain = 3;

  float *object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyFloatObj,
                                  refCount, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject, refCountRetain));
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, refCountRetain,
                                       hipGraphUserObjectMove));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  HIP_CHECK(hipUserObjectRelease(hObject, refCount + refCountRetain));
  HIP_CHECK(hipGraphReleaseUserObject(graph, hObject, refCountRetain));

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}


/**
 * Negative Test for API - hipGraphRetainUserObject
 1) Pass graph as nullptr
 2) Pass User Object as nullptr
 3) Pass initialRefcount as 0
 4) Pass initialRefcount as INT_MAX
 5) Pass flag as 0
 6) Pass flag as INT_MAX
 */

TEST_CASE("Unit_hipGraphRetainUserObject_Negative") {
  hipError_t ret;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float *object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyFloatObj,
                                  1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("Pass graph as nullptr") {
    ret = hipGraphRetainUserObject(nullptr, hObject, 1,
                                    hipGraphUserObjectMove);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass User Object as nullptr") {
    ret = hipGraphRetainUserObject(graph, nullptr, 1,
                                    hipGraphUserObjectMove);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as 0") {
    ret = hipGraphRetainUserObject(graph, hObject, 0,
                                    hipGraphUserObjectMove);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    ret = hipGraphRetainUserObject(graph, hObject, INT_MAX,
                                    hipGraphUserObjectMove);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass flag as 0") {
    ret = hipGraphRetainUserObject(graph, hObject, 1, 0);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass flag as INT_MAX") {
    ret = hipGraphRetainUserObject(graph, hObject, 1, INT_MAX);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Negative Test for API - hipGraphReleaseUserObject
 1) Pass graph as nullptr
 2) Pass User Object as nullptr
 3) Pass initialRefcount as 0
 4) Pass initialRefcount as INT_MAX
 */

TEST_CASE("Unit_hipGraphReleaseUserObject_Negative") {
  hipError_t ret;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float *object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                  destroyFloatObj,
                                  1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 1,
                                       hipGraphUserObjectMove));

  SECTION("Pass graph as nullptr") {
    ret = hipGraphReleaseUserObject(nullptr, hObject, 1);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass User Object as nullptr") {
    ret = hipGraphReleaseUserObject(graph, nullptr, 1);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as 0") {
    ret = hipGraphReleaseUserObject(graph, hObject, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    ret = hipGraphReleaseUserObject(graph, hObject, INT_MAX);
    REQUIRE(hipSuccess == ret);
  }
  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphRetainUserObject_Negative_Basic") {
  hipError_t ret;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float *object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                destroyFloatObj,
                                1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  // Retain graph object with reference count 2
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 2,
                                     hipGraphUserObjectMove));

  // Release graph object with reference count more than 2
  ret = hipGraphReleaseUserObject(graph, hObject, 4);
  REQUIRE(hipSuccess == ret);

  // Again Retain graph object with reference count 8
  ret = hipGraphRetainUserObject(graph, hObject, 8,
                                 hipGraphUserObjectMove);
  REQUIRE(hipSuccess == ret);
  // Release graph object with reference count 1
  ret = hipGraphReleaseUserObject(graph, hObject, 1);
  REQUIRE(hipSuccess == ret);

  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphRetainUserObject_Negative_Null_Object") {
  hipError_t ret;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float *object = nullptr;  // this is used for Null_Object test
  hipUserObject_t hObject;

  HIP_CHECK(hipUserObjectCreate(&hObject, object,
                                destroyFloatObj,
                                1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  // Retain graph object with reference count 2
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 2,
                                     hipGraphUserObjectMove));

  // Release graph object with reference count more than 2
  ret = hipGraphReleaseUserObject(graph, hObject, 4);
  REQUIRE(hipSuccess == ret);

  // Again Retain graph object with reference count 8
  ret = hipGraphRetainUserObject(graph, hObject, 8, 0);
  REQUIRE(hipSuccess == ret);
  // Release graph object with reference count 1
  ret = hipGraphReleaseUserObject(graph, hObject, 1);
  REQUIRE(hipSuccess == ret);

  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}

