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

# Try to find ROCR (Radeon Open Compute Runtime)
#
# Once found, this will define:
#   - ROCR_FOUND     - ROCR status (found or not found)
#   - ROCR_INCLUDES  - Required ROCR include directories
#   - ROCR_LIBRARIES - Required ROCR libraries
find_path(FIND_ROCR_INCLUDES hsa.h HINTS /opt/rocm/include /opt/rocm/hsa/include PATH_SUFFIXES hsa)
find_library(FIND_ROCR_LIBRARIES hsa-runtime64 HINTS /opt/rocm/lib /opt/rocm/hsa/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCR DEFAULT_MSG
                                  FIND_ROCR_INCLUDES FIND_ROCR_LIBRARIES)
mark_as_advanced(FIND_ROCR_INCLUDES FIND_ROCR_LIBRARIES)

set(ROCR_INCLUDES ${FIND_ROCR_INCLUDES})
set(ROCR_LIBRARIES ${FIND_ROCR_LIBRARIES})
