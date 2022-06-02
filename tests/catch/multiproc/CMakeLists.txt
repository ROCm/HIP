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
    hipMemCoherencyTstMProc.cc
    hipIpcEventHandle.cc
)

# the last argument linker libraries is required for this test but optional to the function
hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})
