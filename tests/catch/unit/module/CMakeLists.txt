set(TEST_SRC
    hip_module_common.cc
    hipModuleGetGlobal.cc
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code
                   COMMAND ${CMAKE_CXX_COMPILER} --genco --std=c++17 ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc -o get_global_test_module.code 
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc)   

add_custom_target(get_global_test_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code)

if(HIP_PLATFORM MATCHES "amd")
    set(RTCLIB "hiprtc")
else()
    set(RTCLIB "nvrtc")
endif()
hip_add_exe_to_target(NAME ModuleTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${RTCLIB})
add_dependencies(MemoryTest get_global_test_module)