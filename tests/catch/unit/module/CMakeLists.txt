set(TEST_SRC
)

hip_add_exe_to_target(NAME ModuleTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)