cmake_minimum_required (VERSION 2.6)

project (launch_bounds)

include_directories( ${HIPTEST_SOURCE_DIR} )

make_hip_executable (hip_launch_bounds hip_launch_bounds.cpp)
make_test(hip_launch_bounds " ")
