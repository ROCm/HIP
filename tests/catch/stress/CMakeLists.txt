add_custom_target(stress_test COMMAND "${CMAKE_CTEST_COMMAND}" -R "Stress_")

add_subdirectory(memory)
add_subdirectory(module)
if(HIP_PLATFORM MATCHES "amd")
add_subdirectory(printf)
add_subdirectory(stream)
endif()
