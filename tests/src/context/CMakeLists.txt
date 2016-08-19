cmake_minimum_required (VERSION 2.6)

# Functions for kernel attributes (grid_launch, __launch_bounds__, etc)
project (kernel)

include_directories( ${HIPTEST_SOURCE_DIR} )

build_hip_executable_libcpp (hipCtx_simple hipCtx_simple.cpp)
make_test(hipCtx_simple " " )
