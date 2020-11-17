#project("hip")
cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-keep-memory -Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all -Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hip_hcc.map.in")

if(CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared-libasan")
endif()

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hip)

find_package(PythonInterp REQUIRED)

add_definitions(-D__HIP_ROCclr__ -D__HIP_PLATFORM_HCC__ -DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DCL_TARGET_OPENCL_VERSION=220 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER -DUSE_PROF_API)

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

find_package(ROCclr REQUIRED CONFIG
  PATHS
    /opt/rocm
    /opt/rocm/rocclr)

#############################
# Profiling API support
#############################
# Generate profiling API macros/structures header
set(PROF_API_STR "${PROJECT_BINARY_DIR}/include/hip/hcc_detail/hip_prof_str.h")
set(PROF_API_HDR "${PROJECT_SOURCE_DIR}/include/hip/hcc_detail/hip_runtime_api.h")
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

add_definitions(-DUSE_COMGR_LIBRARY -DCOMGR_DYN_DLL)


find_package(hsa-runtime64 REQUIRED CONFIG
   PATHS
     /opt/rocm/lib
   PATH_SUFFIXES
     cmake/hsa-runtime64
)

find_package(amd_comgr REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/amd_comgr
    lib/cmake/amd_comgr)

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
    execute_process(COMMAND sh -c "${CMAKE_CURRENT_SOURCE_DIR}/../bin/hip_embed_pch.sh ${PROJECT_BINARY_DIR}/include ${PROJECT_SOURCE_DIR}/include ${LLVM_DIR} ${HSA_PATH}" COMMAND_ECHO STDERR RESULT_VARIABLE EMBED_PCH_RC)
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
      /opt/rocm/roctracer
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
target_link_libraries(host INTERFACE hip::amdhip64)

add_library(device INTERFACE)
target_link_libraries(device INTERFACE host)

# Short-Term solution for pre-compiled headers for online compilation
if(__HIP_ENABLE_PCH)
  target_sources(amdhip64 PRIVATE ${CMAKE_BINARY_DIR}/hip_pch.o)
endif()

# TODO: we may create host_static and device_static to let app
# link amdhip64_static

# FIXME: Linux convention is to create static library with same base
# filename.
if(${BUILD_SHARED_LIBS})
    target_link_libraries(amdhip64 PRIVATE amdrocclr_static Threads::Threads dl hsa-runtime64::hsa-runtime64)

    add_custom_command(TARGET amdhip64 POST_BUILD COMMAND
        ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/.hipInfo ${PROJECT_BINARY_DIR}/lib/.hipInfo)
    add_custom_command(TARGET amdhip64 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

    INSTALL(PROGRAMS $<TARGET_FILE:amdhip64> DESTINATION lib COMPONENT MAIN)
else()
    target_link_libraries(amdhip64 PRIVATE Threads::Threads dl hsa-runtime64::hsa-runtime64 amd_comgr)
    # combine objects of vid and hip into amdhip64_static
    add_custom_target(
       amdhip64_static_combiner
       ALL
       COMMAND rm -rf  static_lib_temp && mkdir static_lib_temp && cd static_lib_temp # Create temp folder to contain *.o
       COMMAND ${CMAKE_AR} -x $<TARGET_FILE:amdrocclr_static> # Extract *.o from amdrocclr_static
       COMMAND ${CMAKE_AR} -rcs $<TARGET_FILE:amdhip64> *.o # Append *.o to amdhip64_static
       COMMAND cd .. && rm -rf  static_lib_temp # Remove temp folder
       DEPENDS amdhip64 amdrocclr_static # To make sure this is the last step
       COMMENT "Combining static libs into amdhip64_static"
    )
    INSTALL(PROGRAMS $<TARGET_FILE:amdhip64> DESTINATION lib COMPONENT MAIN)
endif()


INSTALL(TARGETS amdhip64 host device EXPORT hip-targets DESTINATION ${LIB_INSTALL_DIR})
INSTALL(EXPORT hip-targets DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR} NAMESPACE hip::)

