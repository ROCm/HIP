# Common Tests - Test independent of all platforms
set(TEST_SRC
    memcpy.cc
    hipMemcpyMThreadMSize.cc
)

hip_add_exe_to_target(NAME memory
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test)
