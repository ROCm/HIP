cmake_minimum_required (VERSION 2.6)

# Functions for kernel attributes (grid_launch, __launch_bounds__, etc)
project (kernel)

include_directories( ${HIPTEST_SOURCE_DIR} )

build_hip_executable_libcpp (hipLanguageExtensions hipLanguageExtensions.cpp)
make_test(hipLanguageExtensions " " )

build_hip_executable (hipGridLaunch hipGridLaunch.cpp)
make_test(hipGridLaunch " " )

build_hip_executable (launch_bounds launch_bounds.cpp)
make_test(launch_bounds " ")

