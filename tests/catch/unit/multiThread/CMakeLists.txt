# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipMultiThreadDevice.cc
  hipMultiThreadStreams1.cc
  hipMultiThreadStreams2.cc
)

if (UNIX)
  set(__LINKER_LIBS__ pthread)
endif()

hip_add_exe_to_target(NAME MultiThreadTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${__LINKER_LIBS__})
