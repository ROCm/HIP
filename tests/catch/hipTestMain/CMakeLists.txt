if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
    add_definitions(-DHT_LOG_ENABLE)
endif()

add_executable(UnitTests EXCLUDE_FROM_ALL main.cc hip_test_context.cc)
set_property(TARGET UnitTests PROPERTY CXX_STANDARD 17)

target_link_libraries(UnitTests PRIVATE DeviceLibs
                                        MemoryTest
                                        Kernels
                                        stdc++fs)

# Add AMD Only Tests
if(HIP_PLATFORM MATCHES "amd")
    target_link_libraries(UnitTests PRIVATE RTC)
endif()

catch_discover_tests(UnitTests PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
add_dependencies(build_tests UnitTests)

# Add Multiproc tests as seperate binary
if(UNIX)
  add_executable(MultiProcTests EXCLUDE_FROM_ALL main.cc hip_test_context.cc)
  set_property(TARGET MultiProcTests PROPERTY CXX_STANDARD 17)
  target_link_libraries(MultiProcTests PRIVATE MultiProc
                                          stdc++fs)
  catch_discover_tests(MultiProcTests PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
  add_dependencies(build_tests MultiProcTests)
endif()
