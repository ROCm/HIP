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
    hipMallocConcurrencyMproc.cc
    hipMemCoherencyTstMProc.cc
    hipIpcEventHandle.cc
    hipIpcMemAccessTest.cc
    deviceAllocationMproc.cc
    hipNoGpuTsts.cc
)

add_custom_target(dummy_kernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${CMAKE_CURRENT_SOURCE_DIR}/dummy_kernel.cpp -o ${CMAKE_CURRENT_BINARY_DIR}/../multiproc/dummy_kernel.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# the last argument linker libraries is required for this test but optional to the function
if(HIP_PLATFORM MATCHES "nvidia")
hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})
endif()
add_dependencies(build_tests dummy_kernel.code)

