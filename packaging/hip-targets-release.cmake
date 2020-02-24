#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

#get_filename_component cannot resolve the symlinks if called from /opt/rocm/lib/hip
#and do three level up again
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_IMPORT_PREFIX "${_DIR}/../../../" REALPATH)

if(NOT HIP_RUNTIME STREQUAL "vdi")
# Import target "hip::hip_hcc_static" for configuration "Release"
    set_property(TARGET hip::hip_hcc_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
  if(HIP_COMPILER STREQUAL "clang")
    set_target_properties(hip::hip_hcc_static PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhip_hcc_static.a"
    )
  else()
    set_target_properties(hip::hip_hcc_static PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
    IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "hc_am"
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhip_hcc_static.a"
    )
  endif()
  list(APPEND _IMPORT_CHECK_TARGETS hip::hip_hcc_static )
  list(APPEND _IMPORT_CHECK_FILES_FOR_hip::hip_hcc_static "${_IMPORT_PREFIX}/lib/libhip_hcc_static.a" )

endif()
# Import target "hip::hip_hcc" for configuration "Release"
set_property(TARGET hip::hip_hcc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
if(HIP_COMPILER STREQUAL "clang")
  if(HIP_RUNTIME STREQUAL "vdi")
    set_target_properties(hip::amdhip64 PROPERTIES
      IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libamdhip64.so"
      IMPORTED_SONAME_RELEASE "libamdhip64.so")
  else()
    set_target_properties(hip::hip_hcc PROPERTIES
      IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhip_hcc.so"
      IMPORTED_SONAME_RELEASE "libhip_hcc.so")
  endif()
else()
  set_target_properties(hip::hip_hcc PROPERTIES
    IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "hcc::hccrt;hcc::hc_am"
    IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhip_hcc.so"
    IMPORTED_SONAME_RELEASE "libhip_hcc.so")
endif()

if(NOT HIP_RUNTIME STREQUAL "vdi")
  list(APPEND _IMPORT_CHECK_TARGETS hip::hip_hcc )
  list(APPEND _IMPORT_CHECK_FILES_FOR_hip::hip_hcc "${_IMPORT_PREFIX}/lib/libhip_hcc.so" )
else()
  list(APPEND _IMPORT_CHECK_TARGETS hip::amdhip64)
  list(APPEND _IMPORT_CHECK_FILES_FOR_hip::amdhip64 "${_IMPORT_PREFIX}/lib/libamdhip64.so" )
endif()
# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
