# AMD Tests
set(TEST_SRC
    printfFlags.cc
    printfSpecifiers.cc
)


if(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME printfTests
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    PROPERTY CXX_STANDARD 17)
elseif (HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME printfTests
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    COMPILE_OPTIONS -std=c++17)
elseif (HIP_PLATFORM MATCHES "spirv")
hip_add_exe_to_target(NAME printfTests
                  TEST_SRC ${TEST_SRC}
                  TEST_TARGET_NAME build_tests
                  PROPERTY CXX_STANDARD 17)
endif()

# Standalone exes
add_executable(printfFlags EXCLUDE_FROM_ALL printfFlags_exe.cc)
add_executable(printfSpecifiers EXCLUDE_FROM_ALL printfSpecifiers_exe.cc)

add_dependencies(printfTests printfFlags)
add_dependencies(printfTests printfSpecifiers)
