if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
    add_definitions(-DHT_LOG_ENABLE)
endif()

add_executable(UnitTests EXCLUDE_FROM_ALL main.cc hip_test_context.cc)
if(HIP_PLATFORM MATCHES "amd")
    set_property(TARGET UnitTests PROPERTY CXX_STANDARD 17)
else()
    target_compile_options(UnitTests PUBLIC -std=c++17)
endif()

target_link_libraries(UnitTests PRIVATE UnitDeviceTests
                                        MemoryTest
                                        StreamTest
                                        EventTest
                                        OccupancyTest
                                        DeviceTest
                                        RTC
                                        stdc++fs)

if(HIP_PLATFORM MATCHES "nvidia")
    target_link_libraries(UnitTests PRIVATE nvrtc)
endif()

catch_discover_tests(UnitTests PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

# ABM exe
add_executable(ABMTests EXCLUDE_FROM_ALL main.cc hip_test_context.cc)
if(HIP_PLATFORM MATCHES "amd")
    set_property(TARGET ABMTests PROPERTY CXX_STANDARD 17)
else()
    target_compile_options(ABMTests PUBLIC -std=c++17)
endif()

target_link_libraries(ABMTests PRIVATE ABMAddKernels
                                       stdc++fs)

catch_discover_tests(ABMTests PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

add_dependencies(build_tests UnitTests ABMTests)

# Add Multiproc tests as seperate binary
if(UNIX)
    add_executable(MultiProcTests EXCLUDE_FROM_ALL main.cc hip_test_context.cc)

    if(HIP_PLATFORM MATCHES "amd")
        set_property(TARGET MultiProcTests PROPERTY CXX_STANDARD 17)
    else()
        target_compile_options(MultiProcTests PUBLIC -std=c++17)
    endif()

    target_link_libraries(MultiProcTests PRIVATE MultiProc
                                          stdc++fs)
    catch_discover_tests(MultiProcTests PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
    add_dependencies(build_tests MultiProcTests)
endif()

add_executable(StressTest EXCLUDE_FROM_ALL main.cc hip_test_context.cc)
add_custom_target(build_stress_test)
if(HIP_PLATFORM MATCHES "amd")
    set_property(TARGET StressTest PROPERTY CXX_STANDARD 17)
else()
    target_compile_options(StressTest PUBLIC -std=c++17)
endif()
if(HIP_PLATFORM MATCHES "amd")
target_link_libraries(StressTest PRIVATE printf stream)
endif()
target_link_libraries(StressTest PRIVATE memory stdc++fs)
add_dependencies(build_stress_test StressTest)
add_custom_target(stress_test COMMAND StressTest)

# Space Specifiers/Qualifiers exe
add_executable(TypeQualifierTests EXCLUDE_FROM_ALL main.cc hip_test_context.cc)
if(HIP_PLATFORM MATCHES "amd")
    set_property(TARGET TypeQualifierTests PROPERTY CXX_STANDARD 17)
else()
    target_compile_options(TypeQualifierTests PUBLIC -std=c++17)
endif()
target_link_libraries(TypeQualifierTests PRIVATE TypeQualifiers stdc++fs)

catch_discover_tests(TypeQualifierTests PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
add_dependencies(build_tests TypeQualifierTests)
