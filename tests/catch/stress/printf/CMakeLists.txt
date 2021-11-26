# Common Tests - Test independent of all platforms
set(TEST_SRC
    Stress_printf_ComplexKernels.cc
    Stress_printf_SimpleKernels.cc
)

hip_add_exe_to_target(NAME printf
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test)
