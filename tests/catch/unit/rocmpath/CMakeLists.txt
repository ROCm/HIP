# Copyright (c) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
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

set(TEST_SRC
    hipOption_rocm_path.cc
)

# get env var ROCM_PATH
# save ROCM_PATH
set(SAVED_ROCM_PATH $ENV{ROCM_PATH})

# set ROCM_PATH to something wrong
#set(ENV{ROCM_PATH} "/opt/bad_dirr")
#message(STATUS "DAVE ========================== setting ROCM_PATH to [$ENV{ROCM_PATH}]")
#message(STATUS "DAVE: SAVED_ROCM_PATH: ${SAVED_ROCM_PATH}")

# build test with --rocm-path to saved ROCM_PATH
# set_source_files_properties(hipOption_rocm_path.cc PROPERTIES COMPILE_FLAGS -std=c++97)
set_source_files_properties(hipOption_rocm_path.cc PROPERTIES ENVIRONMENT_MODIFICATION "ROCM_PATH=bad-dir")


#add_custom_target(RocmPathTest COMMAND set ROCM_PATH=bad-dir; ${CMAKE_CXX_COMPILER}  ${CMAKE_CURRENT_SOURCE_DIR}/hipOption_rocm_path.cc )


#set_source_files_properties(hipOption_rocm_path.cc PROPERTIES ENVIRONMENT ROCM_PATH="/opt/bad-dir" COMPILE_FLAGS --rocm-path=${SAVED_ROCM_PATH} )
# set_source_files_properties(hipOption_rocm_path.cc PROPERTIES ENVIRONMENT ROCM_PATH=/opt/bad-dir)
# set_property(TEST hipOption_rocm_path APPEND PROPERTY ENVIRONMENT ROCM_PATH="/opt/bad-dir")

hip_add_exe_to_target(NAME UnitOptionsTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

	      #	      set_tests_properties(UnitOptionsTests PROPERTIES ENVIRONMENT_MODIFICATION "ROCM_PATH=bad-dir")

# add_dependencies(build_tests RocmPathTest)

# check for success

# set ROCM_PATH back to original
# set($ENV{ROCM_PATH} SAVED_ROCM_PATH)


