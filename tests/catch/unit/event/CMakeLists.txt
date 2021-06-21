# Common Tests - Test independent of all platforms
set(TEST_SRC
    Unit_hipEvent_Negative.cc
    Unit_hipEvent.cc
    Unit_hipEventElapsedTime.cc
    Unit_hipEventRecord.cc
    Unit_hipEventIpc.cc
)

# Create shared lib of all tests
add_library(EventTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests EventTest)
