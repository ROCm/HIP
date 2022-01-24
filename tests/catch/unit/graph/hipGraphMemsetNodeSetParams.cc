#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/* Test verifies hipGraphMemsetNodeSetParams API invalid params scenarios.
 */
TEST_CASE("Unit_hipGraphMemsetNodeSetParams_InvalidParams") {
  hipError_t ret;
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphNode_t memsetNode;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  char *devData;
  HIP_CHECK(hipMalloc(&devData, 1024));
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devData);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);

  memsetParams.width = 1024;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                  &memsetParams));

  SECTION("Pass element size other than 1, 2, or 4") {
    memsetParams.elementSize = 9;
    ret = hipGraphMemsetNodeSetParams(memsetNode, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass height as zero or negative") {
    memsetParams.elementSize = 2;
    memsetParams.height = 0;
    ret = hipGraphMemsetNodeSetParams(memsetNode, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}