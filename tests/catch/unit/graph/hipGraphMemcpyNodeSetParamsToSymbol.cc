#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <limits>
#define SIZE 256

__device__ int globalIn[SIZE], globalOut[SIZE];
__device__ __constant__ int globalConst[SIZE];


/* This testcase verifies negative scenarios of
   hipGraphMemcpyNodeSetParamsToSymbol API */
TEST_CASE("Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipGraph_t graph;
  hipError_t ret;
  hipGraphNode_t memcpyToSymbolNode, memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);

  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            HIP_SYMBOL(globalIn),
                                            A_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));

  SECTION("Pass GraphNode as nullptr") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(nullptr,
                                              HIP_SYMBOL(globalIn),
                                              B_d, Nbytes, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass symbol ptr as nullptr") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              nullptr,
                                              B_d, Nbytes, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidSymbol == ret);
  }
  SECTION("Pass src ptr as nullptr") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              HIP_SYMBOL(globalIn),
                                              nullptr, Nbytes, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass count as zero") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              HIP_SYMBOL(globalIn),
                                              B_d, 0, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  SECTION("Pass count more than allocated size for source and dstn ptr") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              HIP_SYMBOL(globalIn),
                                              B_d, Nbytes+8, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass offset+count greater than allocated size") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              HIP_SYMBOL(globalIn),
                                              B_d, Nbytes, 10,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass same symbol pointer as source ptr and destination ptr") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              HIP_SYMBOL(globalIn),
                                              HIP_SYMBOL(globalIn),
                                              Nbytes, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass 2 different symbol pointer as source ptr and dstn ptr") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              HIP_SYMBOL(globalIn),
                                              HIP_SYMBOL(globalOut),
                                              Nbytes, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Copy from host ptr to device ptr but pass kind as different") {
    ret = hipGraphMemcpyNodeSetParamsToSymbol(memcpyToSymbolNode,
                                              HIP_SYMBOL(globalIn),
                                              A_h,
                                              Nbytes, 0,
                                              hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph));
}