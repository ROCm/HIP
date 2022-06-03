# Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# hipcc.bat fails to qualify as a valid compiler for CMAKE_CXX_COMPILER_ID = ROCMClang
# so  the simple compiler test is skipped and forced to use hipcc.bat as compiler
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

project(hipInfo)

cmake_minimum_required(VERSION 3.10)

# flag is set to ON in compute build for windows
option(HIPINFO_INTERNAL_BUILD "Enable building hipInfo from compute" OFF)

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# need to set rocm_path for windows
# since clang and hip are two different folders during build/install step
if (WIN32 AND HIPINFO_INTERNAL_BUILD)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --rocm-path=${HIP_PATH}")
endif()

# Find hip
find_package(hip)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Create the excutable
add_executable(hipInfo hipInfo.cpp)

# Link with HIP
target_link_libraries(hipInfo hip::host)

# Used only when make install is called
# when hipInfo is built as part of compute project
# hipInfo.exe will be installed to install/hip/bin path
if (WIN32 AND HIPINFO_INTERNAL_BUILD)
     install(FILES ${PROJECT_BINARY_DIR}/hipInfo.exe DESTINATION .)
endif()
