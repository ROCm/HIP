set(TEST_SRC
    hip_module_common.cc
    hipModuleLaunchKernel.cc 
)

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC} hipExtModuleLaunchKernel.cc)
endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code
                   COMMAND ${CMAKE_CXX_COMPILER} --genco --std=c++17 ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc -o launch_kernel_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_module.cc)   
add_custom_target(launch_kernel_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/launch_kernel_module.code)

if(HIP_PLATFORM MATCHES "amd")
    set(RTCLIB "hiprtc")
else()
    set(RTCLIB "nvrtc")
endif()
hip_add_exe_to_target(NAME ModuleTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${RTCLIB})
add_dependencies(ModuleTest launch_kernel_module)