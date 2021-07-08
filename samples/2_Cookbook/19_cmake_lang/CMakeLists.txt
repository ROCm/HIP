# The test must be with CMake 3.18 or above which supports LINK_LANGUAGE that is necessary for
# Fortran build.
cmake_minimum_required(VERSION 3.18.0)

project(cmake_lang_test CXX Fortran)

find_package(hip REQUIRED)

add_executable(test_fortran TestFortran.F90)
add_executable(test_cpp MatrixTranspose.cpp)

target_link_libraries(test_cpp PUBLIC hip::device)
# Assuming to build a C/C++-to-Fortran library binding.
target_link_libraries(test_fortran PUBLIC hip::device)
