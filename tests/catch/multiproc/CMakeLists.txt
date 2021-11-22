# Common Tests
set(LINUX_TEST_SRC
    childMalloc.cc
    hipDeviceComputeCapabilityMproc.cc
    hipDeviceGetPCIBusIdMproc.cc
    hipDeviceTotalMemMproc.cc
    hipGetDeviceAttributeMproc.cc
    hipGetDeviceCountMproc.cc
    hipGetDevicePropertiesMproc.cc
    hipSetGetDeviceMproc.cc
    hipIpcMemAccessTest.cc
    hipHostMallocTestsMproc.cc
    hipMallocConcurrencyMproc.cc
)

if(UNIX)
    # Create shared lib of all tests
    add_library(MultiProc SHARED EXCLUDE_FROM_ALL ${LINUX_TEST_SRC})

    target_link_libraries(MultiProc ${CMAKE_DL_LIBS})

    # Add dependency on build_tests to build it on this custom target
    add_dependencies(build_tests MultiProc)
endif()
