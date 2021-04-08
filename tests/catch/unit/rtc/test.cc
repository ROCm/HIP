#include <hip_test_common.hh>

TEST_CASE("cpp17 test") {
  constexpr auto l = []() { return 2 * 10 * 30; };
  REQUIRE(l() == 600);
}
