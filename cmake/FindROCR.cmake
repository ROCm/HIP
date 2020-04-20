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
