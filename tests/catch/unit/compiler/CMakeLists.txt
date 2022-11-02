# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipClassKernel.cc
)

hip_add_exe_to_target(NAME CompilerTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
