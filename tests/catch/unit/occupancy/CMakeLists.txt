# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipOccupancyMaxActiveBlocksPerMultiprocessor.cc
  hipOccupancyMaxPotentialBlockSize.cc
  hipModuleOccupancyMaxPotentialBlockSize.cc
  hipModuleOccupancyMaxPotentialBlockSizeWithFlags.cc
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.cc
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.cc
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/simple_kernel.code
                   COMMAND ${CMAKE_CXX_COMPILER} --genco --std=c++17 ${CMAKE_CURRENT_SOURCE_DIR}/simple_kernel.cc -o simple_kernel.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/simple_kernel.cc)

add_custom_target(simple_kernel ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/simple_kernel.code)

hip_add_exe_to_target(NAME OccupancyTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

add_dependencies(OccupancyTest simple_kernel)