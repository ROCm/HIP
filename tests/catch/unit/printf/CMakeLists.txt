# AMD Tests
set(TEST_SRC
    printfFlags.cc
    printfSpecifiers.cc
)

# Create shared lib of all tests
add_library(printfTests SHARED EXCLUDE_FROM_ALL ${TEST_SRC})
if(HIP_PLATFORM MATCHES "amd")
    set_property(TARGET printfTests PROPERTY CXX_STANDARD 17)
else()
    target_compile_options(printfTests PUBLIC -std=c++17)
endif()

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests printfTests)
