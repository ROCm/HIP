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
