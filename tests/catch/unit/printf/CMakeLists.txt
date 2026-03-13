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

if(NOT STANDALONE_TESTS)
  add_executable(printfFlags EXCLUDE_FROM_ALL printfFlags_exe.cc)
  add_executable(printfSpecifiers EXCLUDE_FROM_ALL printfSpecifiers_exe.cc)

  add_dependencies(printfTests printfFlags)
  add_dependencies(printfTests printfSpecifiers)
else()
  # When STANDALONE_TESTS, the test runner is also named printfFlags. Build the
  # kernel exe with _proc suffix so SpawnProc can find and run it.
  add_executable(printfFlags_proc EXCLUDE_FROM_ALL printfFlags_exe.cc)
  add_executable(printfSpecifiers_proc EXCLUDE_FROM_ALL printfSpecifiers_exe.cc)
  set_target_properties(printfFlags_proc printfSpecifiers_proc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(printfTests printfFlags_proc)
  add_dependencies(printfTests printfSpecifiers_proc)
  target_compile_definitions(printfTests PRIVATE HIP_STANDALONE_PRINTF_PROC=1)
endif()