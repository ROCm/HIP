# Common Tests - Test independent of all platforms
set(TEST_SRC
    Stress_hipStreamCreate.cc
)

hip_add_exe_to_target(NAME stream
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test)
