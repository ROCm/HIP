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

add_custom_target(module_kernels.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${CMAKE_CURRENT_SOURCE_DIR}/module_kernels.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/module_kernels.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

hip_add_exe_to_target(NAME ModuleTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
add_dependencies(build_tests module_kernels.code)
