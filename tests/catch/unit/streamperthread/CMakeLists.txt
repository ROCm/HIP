# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipStreamPerThread_Basic.cc
  hipStreamPerThread_Event.cc
  hipStreamPerThread_MultiThread.cc
  hipStreamPerThread_DeviceReset.cc
)

# Create shared lib of all tests
add_library(StreamPerThreadTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests StreamPerThreadTest)
