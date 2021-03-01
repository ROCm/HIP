# Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All Rights Reserved.
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

# Try to find ROCT (Radeon Open Compute Thunk)
#
# Once found, this will define:
#   - ROCT_FOUND     - ROCT status (found or not found)
#   - ROCT_INCLUDES  - Required ROCT include directories
#   - ROCT_LIBRARIES - Required ROCT libraries
find_path(FIND_ROCT_INCLUDES hsakmt.h HINTS /opt/rocm/include)
find_library(FIND_ROCT_LIBRARIES hsakmt HINTS /opt/rocm/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCT DEFAULT_MSG
                                  FIND_ROCT_INCLUDES FIND_ROCT_LIBRARIES)
mark_as_advanced(FIND_ROCT_INCLUDES FIND_ROCT_LIBRARIES)

set(ROCT_INCLUDES ${FIND_ROCT_INCLUDES})
set(ROCT_LIBRARIES ${FIND_ROCT_LIBRARIES})
