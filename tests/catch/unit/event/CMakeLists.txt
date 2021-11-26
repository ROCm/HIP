# Common Tests - Test independent of all platforms
set(TEST_SRC
    Unit_hipEvent_Negative.cc
    Unit_hipEvent.cc
    Unit_hipEventElapsedTime.cc
    Unit_hipEventRecord.cc
    Unit_hipEventIpc.cc
)

hip_add_exe_to_target(NAME EventTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
