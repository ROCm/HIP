# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipChooseDevice.cc
    hipDeviceComputeCapability.cc
    hipDeviceGetByPCIBusId.cc
    hipDeviceGetLimit.cc
    hipDeviceGetName.cc
    hipDeviceGetPCIBusId.cc
    hipDeviceSetGetCacheConfig.cc
    hipDeviceSynchronize.cc
    hipDeviceTotalMem.cc
    hipGetDeviceAttribute.cc
    hipGetDeviceCount.cc
    hipGetDeviceProperties.cc
    hipRuntimeGetVersion.cc
    hipGetSetDeviceFlags.cc
    hipSetGetDevice.cc
    hipDeviceGetUuid.cc
    hipDeviceGetP2PAttribute.cc
    hipDeviceSetGetSharedMemConfig.cc
)

set_source_files_properties(hipGetDeviceCount.cc PROPERTIES COMPILE_FLAGS -std=c++17)
set_source_files_properties(hipDeviceGetP2PAttribute.cc PROPERTIES COMPILE_FLAGS -std=c++17)

add_executable(getDeviceCount EXCLUDE_FROM_ALL getDeviceCount_exe.cc)
add_executable(hipDeviceGetP2PAttribute EXCLUDE_FROM_ALL hipDeviceGetP2PAttribute_exe.cc)

hip_add_exe_to_target(NAME DeviceTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++14)

add_dependencies(DeviceTest getDeviceCount)
add_dependencies(DeviceTest hipDeviceGetP2PAttribute)
