#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hip::hip_hcc_static" for configuration "Release"
set_property(TARGET hip::hip_hcc_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hip::hip_hcc_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "hc_am"
  IMPORTED_LOCATION_RELEASE "/opt/rocm/hip/lib/libhip_hcc_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS hip::hip_hcc_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hip::hip_hcc_static "/opt/rocm/hip/lib/libhip_hcc_static.a" )

# Import target "hip::hip_hcc" for configuration "Release"
set_property(TARGET hip::hip_hcc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hip::hip_hcc PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "hcc::hccrt;hcc::hc_am"
  IMPORTED_LOCATION_RELEASE "/opt/rocm/hip/lib/libhip_hcc.so"
  IMPORTED_SONAME_RELEASE "libhip_hcc.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS hip::hip_hcc )
list(APPEND _IMPORT_CHECK_FILES_FOR_hip::hip_hcc "/opt/rocm/hip/lib/libhip_hcc.so" )

# Import target "hip::hip_device" for configuration "Release"
set_property(TARGET hip::hip_device APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hip::hip_device PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/opt/rocm/hip/lib/libhip_device.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS hip::hip_device )
list(APPEND _IMPORT_CHECK_FILES_FOR_hip::hip_device "/opt/rocm/hip/lib/libhip_device.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
