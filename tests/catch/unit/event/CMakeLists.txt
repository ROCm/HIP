# Common Tests - Test independent of all platforms
set(TEST_SRC
    Unit_hipEvent_Negative.cc
    Unit_hipEvent.cc
    Unit_hipEventElapsedTime.cc
)

# skipped for windows due to duplicate symbols
if(UNIX)
    set(TEST_SRC ${TEST_SRC}
                 Unit_hipEventRecord.cc
                 Unit_hipEventIpc.cc)
endif()

# Create shared lib of all tests
add_library(EventTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests EventTest)
