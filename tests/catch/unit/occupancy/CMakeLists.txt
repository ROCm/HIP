# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipOccupancyMaxActiveBlocksPerMultiprocessor.cc
  hipOccupancyMaxActiveBlocksPerMultiprocessor_old.cc
  hipOccupancyMaxPotentialBlockSize.cc
  hipOccupancyMaxPotentialBlockSize_old.cc
)

hip_add_exe_to_target(NAME OccupancyTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
