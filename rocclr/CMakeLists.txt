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

#project("hip")
cmake_minimum_required(VERSION 3.5.1)

option(ADDRESS_SANITIZER "Build Address Sanitizer" OFF)
if (ADDRESS_SANITIZER)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    if (BUILD_SHARED_LIBS)
        set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" -fsanitize=address)
        set (CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS}" -shared-libsan)
    else ()
        set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS}" -fsanitize=address)
        set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS}" -static-libsan)
    endif ()
endif ()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-keep-memory -Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all -Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hip_hcc.map.in")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hip)

find_package(PythonInterp REQUIRED)

add_definitions( -D__HIP_PLATFORM_AMD__ -DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DCL_TARGET_OPENCL_VERSION=220 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER -DUSE_PROF_API)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
   (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang"))
  add_definitions(
    # Enabling -Wextra or -pedantic will cause
    # thousands of warnings. Keep things simple for now.
    -Wall
    # This one seems impossible to fix for now.
    # There are hundreds of instances of unused vars/functions
    # throughout the code base.
    -Wno-unused-variable
    -Wno-unused-function)
endif()

set(USE_PROF_API "1")

# Need to add /opt/rocm/llvm to package search path since ROCclr will
# find package amd_comgr and amd_comgr will find package llvm/clang.
# Without this, the system llvm/clang at /usr/local may be found.
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/rocm/llvm")
find_package(ROCclr REQUIRED CONFIG)

#############################
# Profiling API support
#############################
# Generate profiling API macros/structures header
set(PROF_API_STR "${PROJECT_BINARY_DIR}/include/hip/amd_detail/hip_prof_str.h")
set(PROF_API_HDR "${PROJECT_SOURCE_DIR}/include/hip/amd_detail/hip_runtime_api.h")
set(PROF_API_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROF_API_GEN "${CMAKE_CURRENT_SOURCE_DIR}/hip_prof_gen.py")
set(PROF_API_LOG "${PROJECT_BINARY_DIR}/hip_prof_gen.log.txt")

add_custom_command(OUTPUT ${PROF_API_STR}
  COMMAND ${PYTHON_EXECUTABLE} ${PROF_API_GEN} -v -t --priv ${OPT_PROF_API} ${PROF_API_HDR} ${PROF_API_SRC} ${PROF_API_STR}
  OUTPUT_FILE ${PROF_API_LOG}
  DEPENDS ${PROF_API_HDR} ${PROF_API_GEN}
  COMMENT "Generating profiling primitives: ${PROF_API_STR}")

add_custom_target(gen-prof-api-str-header ALL
  DEPENDS ${PROF_API_STR}
  SOURCES ${PROF_API_HDR})

# Enable profiling API
if(NOT DEFINED ROCclr_DIR OR NOT DEFINED LIBOCL_STATIC_DIR OR NOT DEFINED LIBROCclr_STATIC_DIR )
	#	message(FATAL_ERROR "define ROCclr_DIR, LIBOCL_STATIC_DIR\n")

endif()

#APPEND default path for CMAKE_PREFIX_PATH
#User provided will be searched first since defualt path is at end.
#Custom install path can be provided at compile time as cmake parameter(-DCMAKE_PREFIX_PATH="")
#/opt/rocm:default:For amd_comgr,hsa-runtime64
#/opt/rocm/llvm/:default:For llvm/clang pulled in as dependency from hsa/comgr
list( APPEND CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/rocm")

list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" )
set(CMAKE_MODULE_PATH${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

message(STATUS "Hsa runtime found at ${hsa-runtime64_DIR}.")
message(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

find_package(LLVM REQUIRED CONFIG
   PATHS
     /opt/rocm/llvm
   PATH_SUFFIXES
     lib/cmake/llvm)

message(STATUS "llvm found at ${LLVM_DIR}.")

add_library(hip64 OBJECT
 hip_context.cpp
 hip_code_object.cpp
 hip_device.cpp
 hip_device_runtime.cpp
 hip_error.cpp
 hip_event.cpp
 hip_fatbin.cpp
 hip_global.cpp
 hip_hmm.cpp
 hip_memory.cpp
 hip_module.cpp
 hip_peer.cpp
 hip_platform.cpp
 hip_profile.cpp
 hip_stream.cpp
 hip_stream_ops.cpp
 hip_surface.cpp
 hip_texture.cpp
 hip_activity.cpp
 hip_intercept.cpp
 hip_rtc.cpp
 cl_gl.cpp
 cl_lqdflash_amd.cpp
 fixme.cpp
 )
set_target_properties(hip64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(hip64
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/amdocl
    ${ROCR_INCLUDES}
    $<TARGET_PROPERTY:amdrocclr_static,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(hip64
  PRIVATE
   $<TARGET_PROPERTY:amd_comgr,INTERFACE_COMPILE_DEFINITIONS>)

if(ROCclr_FOUND)
  message(STATUS "ROCclr found at ${ROCclr_DIR}")
  target_include_directories(hip64
    PRIVATE
      $<TARGET_PROPERTY:amdrocclr_static,INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(hip64
    PRIVATE
      $<TARGET_PROPERTY:amdrocclr_static,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# Short-Term solution for pre-compiled headers for online compilation
# Enable pre compiled header
if(__HIP_ENABLE_PCH)
    execute_process(COMMAND sh -c "${CMAKE_CURRENT_SOURCE_DIR}/../bin/hip_embed_pch.sh ${PROJECT_BINARY_DIR}/include ${PROJECT_SOURCE_DIR}/include ${LLVM_DIR} ${HSA_PATH}" COMMAND_ECHO STDERR  RESULT_VARIABLE EMBED_PCH_RC)
    if (EMBED_PCH_RC AND NOT EMBED_PCH_RC EQUAL 0)
      message(FATAL_ERROR "Failed to embed PCH")
    endif()
    add_definitions(-D__HIP_ENABLE_PCH)
endif()

# Enable profiling API
if(USE_PROF_API EQUAL 1)
  find_path(PROF_API_HEADER_DIR prof_protocol.h
    HINTS
      ${PROF_API_HEADER_PATH}
    PATHS
      ${ROCM_PATH}/roctracer
    PATH_SUFFIXES
      include/ext)

  if(NOT PROF_API_HEADER_DIR)
    message(WARNING "Profiling API header not found. Disabling roctracer integration. Use -DPROF_API_HEADER_PATH=<path to prof_protocol.h header>")
    else()
      target_compile_definitions(hip64 PUBLIC USE_PROF_API=1)
      target_include_directories(hip64 PUBLIC ${PROF_API_HEADER_DIR})
      message(STATUS "Profiling API: ${PROF_API_HEADER_DIR}")
    endif()
endif()

set_target_properties(
    hip64 PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        )
add_dependencies(hip64 gen-prof-api-str-header)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


if(${BUILD_SHARED_LIBS})

  add_library(amdhip64
      $<TARGET_OBJECTS:hip64>
  )

  set_target_properties(
      amdhip64 PROPERTIES
         VERSION ${HIP_LIB_VERSION_STRING}
         SOVERSION ${HIP_LIB_VERSION_MAJOR}
      )

  set_target_properties(hip64 PROPERTIES PUBLIC_HEADER ${PROF_API_STR})

else()

   add_library(amdhip64 STATIC
      $<TARGET_OBJECTS:hip64>
      )

endif()

set_target_properties(amdhip64 PROPERTIES LINK_FLAGS_RELEASE -s)
set_target_properties(amdhip64 PROPERTIES PUBLIC_HEADER ${PROF_API_STR})
add_library(host INTERFACE)
target_link_libraries(host INTERFACE amdhip64)

add_library(device INTERFACE)
target_link_libraries(device INTERFACE host)

# Short-Term solution for pre-compiled headers for online compilation
if(__HIP_ENABLE_PCH)
  target_sources(amdhip64 PRIVATE ${CMAKE_BINARY_DIR}/hip_pch.o)
endif()

target_link_libraries(amdhip64 PRIVATE amdrocclr_static Threads::Threads dl hsa-runtime64::hsa-runtime64)
add_custom_command(TARGET amdhip64 POST_BUILD COMMAND
    ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/.hipInfo ${PROJECT_BINARY_DIR}/lib/.hipInfo)
add_custom_command(TARGET amdhip64 POST_BUILD COMMAND cp -rf
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/)

INSTALL(PROGRAMS $<TARGET_FILE:amdhip64> DESTINATION lib COMPONENT MAIN)

INSTALL(TARGETS amdhip64 host device EXPORT hip-targets DESTINATION ${LIB_INSTALL_DIR})
INSTALL(EXPORT hip-targets DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR} NAMESPACE hip::)

INSTALL(TARGETS amdhip64 host device EXPORT hip-lang-targets DESTINATION ${LIB_INSTALL_DIR})
INSTALL(EXPORT hip-lang-targets DESTINATION ${CONFIG_LANG_PACKAGE_INSTALL_DIR} NAMESPACE hip-lang::)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/hip-lang-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config.cmake
    INSTALL_DESTINATION ${CONFIG_LANG_PACKAGE_INSTALL_DIR}
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config-version.cmake
    VERSION "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_GITDATE}"
    COMPATIBILITY SameMajorVersion
    )
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config-version.cmake
    DESTINATION
    ${CONFIG_LANG_PACKAGE_INSTALL_DIR}/
    )
