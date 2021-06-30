# Common Tests - Test independent of all platforms
set(TEST_SRC
    saxpy.cc
)

# Set source File properties
set_source_files_properties(saxpy.cc PROPERTIES COMPILE_FLAGS " -std=c++14 ")
set_source_files_properties(test.cc PROPERTIES COMPILE_FLAGS " -std=c++17 ")

# Create shared lib of all tests
add_library(RTC SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests RTC)
