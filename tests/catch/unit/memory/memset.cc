#include <hip_test_common.hh>

TEST_CASE("Unit_hipMemset_4bytes") {
  int* d_a;
  auto res = hipMalloc(&d_a, sizeof(int));
  REQUIRE(res == hipSuccess);
  res = hipMemset(d_a, 0, sizeof(int));
  REQUIRE(res == hipSuccess);
  res = hipFree(d_a);
  REQUIRE(res == hipSuccess);
}

TEST_CASE("Unit_hipMemset_4bytes_hostMem") {
  int* d_a;
  auto res = hipHostMalloc(&d_a, sizeof(int), 0);
  REQUIRE(res == hipSuccess);
  res = hipMemset(d_a, 0, sizeof(int));
  REQUIRE(res == hipSuccess);
  res = hipHostFree(d_a);
  REQUIRE(res == hipSuccess);
}
