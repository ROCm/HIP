#include <hip_test_common.hh>

TEST_CASE("Unit_hipMemset_4bytes") {
  int* d_a;
  auto res = hipMalloc(&d_a, sizeof(int));
  REQUIRE(res == hipSuccess);
  res = hipMemset(d_a, 0, sizeof(int));
  REQUIRE(res == hipSuccess);
  hipFree(d_a);
}

TEST_CASE("Unit_hipMemset_4bytes_hostMem") {
  int* d_a;
  auto res = hipHostMalloc(&d_a, sizeof(int), 0);
  REQUIRE(res == hipSuccess);
  res = hipMemset(d_a, 0, sizeof(int));
  REQUIRE(res == hipSuccess);
  hipHostFree(d_a);
}
