# Common Tests - Test independent of all platforms
set(TEST_SRC
    add.cc
)

foreach(exe_name ${TEST_SRC})
    hip_add_exe_to_target(NAME ${exe_name}
                      TEST_SRC ${exe_name}
                      TEST_TARGET_NAME build_tests)
endforeach()