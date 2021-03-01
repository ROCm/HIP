# Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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

###############################################################################
# Tests.cmake
###############################################################################

# Add tests
include_directories(${HIP_SRC_PATH}/tests/src)
hit_add_directory_recursive(${HIP_CTEST_CONFIG_DEFAULT} ${HIP_SRC_PATH}/tests/src "directed_tests")

# Add unit tests
include_directories(${HIP_SRC_PATH}/tests/unit)
hit_add_directory_recursive(${HIP_CTEST_CONFIG_DEFAULT} ${HIP_SRC_PATH}/tests/unit "unit_tests")

# Add performance tests
include_directories(${HIP_SRC_PATH}/tests/performance)
hit_add_directory_recursive(${HIP_CTEST_CONFIG_PERFORMANCE} ${HIP_SRC_PATH}/tests/performance "performance_tests")

# Add top-level tests to build_tests
add_custom_target(build_tests DEPENDS directed_tests unit_tests)

# Add top-level tests to build performance_tests.
# To build performance tests, just run "make build_perf"
add_custom_target(build_perf DEPENDS performance_tests)

# Add custom target: perf.
# To run performance tests, just run "make perf"
add_custom_target(perf COMMAND "${CMAKE_CTEST_COMMAND}" -C "${HIP_CTEST_CONFIG_PERFORMANCE}" -R "performance_tests/" --verbose)

# Add custom target: check
add_custom_target(check COMMAND "${CMAKE_COMMAND}" --build . --target test DEPENDS build_tests)

# vim: ts=4:sw=4:expandtab:smartindent