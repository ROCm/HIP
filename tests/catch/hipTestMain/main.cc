#define CATCH_CONFIG_RUNNER
#include <hip_test_common.hh>
#include <iostream>

int main(int argc, char** argv) {
  auto& context = TestContext::get(argc, argv);
  if (context.skipTest()) {
    // CTest uses this regex to figure out if the test has been skipped
    std::cout << "HIP_SKIP_THIS_TEST" << std::endl;
    return 0;
  }
  int out = Catch::Session().run(argc, argv);
  TestContext::get().cleanContext();
  return out;
  
}
