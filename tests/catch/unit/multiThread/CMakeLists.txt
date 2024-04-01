# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipMultiThreadDevice.cc
  hipMultiThreadStreams1.cc
  hipMultiThreadStreams2.cc
)

hip_add_exe_to_target(NAME MultiThreadTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
