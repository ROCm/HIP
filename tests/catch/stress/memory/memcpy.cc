#include <hip_test_common.hh>

TEST_CASE("hipMalloc", "DifferentSizes") {
  int* d_a = nullptr;
  SECTION("Size 10") {
    auto res = hipMalloc(&d_a, sizeof(10));
    REQUIRE(res == hipSuccess);
    hipFree(d_a);
    d_a = nullptr;
  }
  SECTION("Size 100") {
    auto res = hipMalloc(&d_a, sizeof(100));
    REQUIRE(res == hipSuccess);
    hipFree(d_a);
    d_a = nullptr;
  }
  SECTION("Size 1000") {
    auto res = hipMalloc(&d_a, sizeof(1000));
    REQUIRE(res == hipSuccess);
    hipFree(d_a);
    d_a = nullptr;
  }
  SECTION("Size 10000") {
    auto res = hipMalloc(&d_a, sizeof(10000));
    REQUIRE(res == hipSuccess);
    hipFree(d_a);
    d_a = nullptr;
  }
  SECTION("Size MAX") {
    auto res = hipMalloc(&d_a, ~(size_t)0);
    REQUIRE(res == hipErrorOutOfMemory);
    d_a = nullptr;
  }
}