# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipStreamPerThread_Basic.cc
  hipStreamPerThread_Event.cc
  hipStreamPerThread_MultiThread.cc
  hipStreamPerThread_DeviceReset.cc
  # error: expected namespace name using namespace cooperative_groups;
  # hipStreamPerThrdTsts.cc
)

hip_add_exe_to_target(NAME StreamPerThreadTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
