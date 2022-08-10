# Common Tests - Test independent of all platforms
set(TEST_SRC
    Stress_deviceAllocationStress.cc
)

hip_add_exe_to_target(NAME devalloc
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test)
