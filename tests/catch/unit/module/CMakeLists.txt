# Common Tests - Test independent of all platforms
if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
hipExtLaunchKernelGGL.cc
hipExtModuleLaunchKernel.cc
hipExtLaunchMultiKernelMultiDevice.cc
hipModuleLaunchKernel.cc
hipFuncSetCacheConfig.cc
hipModuleUnload.cc
hipFuncSetAttribute.cc
hipModuleLoadData.cc
hipFuncSetSharedMemConfig.cc
hipManagedKeyword.cc
hipModuleGetGlobal.cc
hipFuncGetAttributes.cc
hipModule.cc
hipModuleLoadDataMultThreadOnMultGPU.cc
hipModuleLoadDataMultThreaded.cc
hipModuleLoadMultiThreaded.cc
hipModuleLoadUnloadStress.cc
hipModuleNegative.cc
hipModuleOccupancyMaxPotentialBlockSize.cc
hipModuleTexture2dDrv.cc
hipOpenCLCOTest.cc
)
else()
set(TEST_SRC
hipModuleLaunchKernel.cc
hipFuncSetCacheConfig.cc
hipModuleUnload.cc
hipFuncSetAttribute.cc
hipModuleLoadData.cc
hipFuncSetSharedMemConfig.cc
hipManagedKeyword.cc
hipModuleGetGlobal.cc
hipFuncGetAttributes.cc
hipModule.cc
hipModuleLoadDataMultThreadOnMultGPU.cc
hipModuleLoadDataMultThreaded.cc
hipModuleLoadMultiThreaded.cc
hipModuleLoadUnloadStress.cc
hipModuleNegative.cc
hipModuleOccupancyMaxPotentialBlockSize.cc
)
endif()

add_custom_target(module_kernels.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${CMAKE_CURRENT_SOURCE_DIR}/module_kernels.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../../hipTestMain/module_kernels.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
# Create shared lib of all tests
add_library(ModuleTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})
add_dependencies(build_tests ModuleTest module_kernels.code)
