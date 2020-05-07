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

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("-Wno-ignored-attributes")
endif()

set(USE_PROF_API "1")

if(NOT DEFINED LIBROCclr_STATIC_DIR)
   find_path(LIBROCclr_STATIC_DIR
        NAMES libamdrocclr_static.a
        PATHS /opt/rocm/rocclr
        PATH_SUFFIXES lib
   )
endif()

if(NOT DEFINED ROCclr_DIR)
   find_path(ROCclr_DIR
        NAMES top.hpp
        PATH_SUFFIXES include
        PATHS /opt/rocm/rocclr
   )
endif()
message("Found Static rocclr lib:${LIBROCclr_STATIC_DIR} and rocclr includes: ${ROCclr_DIR}")
set(PROF_API_HEADER_PATH ${ROCclr_DIR}/platform)
#############################
# Profiling API support
#############################
# Generate profiling API macros/structures header
# FIXME: This should not be writing to the source directory
set(PROF_API_STR "${PROJECT_BINARY_DIR}/include/hip/hcc_detail/hip_prof_str.h")
set(PROF_API_HDR "${CMAKE_CURRENT_SOURCE_DIR}/../include/hip/hcc_detail/hip_runtime_api.h")
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
if(USE_PROF_API EQUAL 1)
    find_path(PROF_API_HEADER_DIR prof_protocol.h
        HINTS
            ${PROF_API_HEADER_PATH}
        PATHS
            /opt/rocm/roctracer
        PATH_SUFFIXES
            include/ext
    )
    if(NOT PROF_API_HEADER_DIR)
        MESSAGE(WARNING "Profiling API header not found. Disabling roctracer integration. Use -DPROF_API_HEADER_PATH=<path to prof_protocol.h header>")
    else()
        add_definitions(-DUSE_PROF_API=1)
        include_directories(${PROF_API_HEADER_DIR})
        MESSAGE(STATUS "Profiling API: ${PROF_API_HEADER_DIR}")
    endif()
endif()


if(NOT DEFINED ROCclr_DIR OR NOT DEFINED LIBOCL_STATIC_DIR OR NOT DEFINED LIBROCclr_STATIC_DIR )
	#	message(FATAL_ERROR "define ROCclr_DIR, LIBOCL_STATIC_DIR\n")

endif()
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" )
set(CMAKE_MODULE_PATH${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include_directories(${ROCR_INCLUDES})
if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")
   message(STATUS "LLVM includes found ${LLVM_INCLUDES}")
   include_directories(${LLVM_INCLUDES})
endif() # if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/elfio)
include_directories(${CMAKE_SOURCE_DIR}/amdocl)
include_directories(${CMAKE_SOURCE_DIR}/include/hip/hcc_detail/elfio)
include_directories(${ROCclr_DIR})
include_directories(${ROCclr_DIR}/include)
include_directories(${ROCclr_DIR}/compiler/lib)
include_directories(${ROCclr_DIR}/compiler/lib/include)
include_directories(${ROCclr_DIR}/elf/utils/common)
include_directories(${ROCclr_DIR}/elf/utils/libelf)
add_definitions(-DUSE_COMGR_LIBRARY -DCOMGR_DYN_DLL)
    find_package(amd_comgr REQUIRED CONFIG
            PATHS
                /opt/rocm/
            PATH_SUFFIXES
                cmake/amd_comgr
                lib/cmake/amd_comgr
    )
    MESSAGE(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

include_directories("$<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>")

add_definitions(-DBSD_LIBELF)

add_library(hip64 OBJECT
 hip_context.cpp
 hip_device.cpp
 hip_device_runtime.cpp
 hip_error.cpp
 hip_event.cpp
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

set_target_properties(
    hip64 PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        )
add_dependencies(hip64 gen-prof-api-str-header)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include(${LIBROCclr_STATIC_DIR}/amdrocclr_staticTargets.cmake)

add_library(amdhip64 SHARED
    $<TARGET_OBJECTS:hip64>
    )

add_library(amdhip64_static STATIC
    $<TARGET_OBJECTS:hip64>
    )

set_target_properties(
    amdhip64 PROPERTIES
        VERSION ${HIP_LIB_VERSION_STRING}
        SOVERSION ${HIP_LIB_VERSION_MAJOR}
        )

set_target_properties(hip64 PROPERTIES PUBLIC_HEADER ${PROF_API_STR})
set_target_properties(amdhip64 PROPERTIES PUBLIC_HEADER ${PROF_API_STR})
set_target_properties(amdhip64_static PROPERTIES PUBLIC_HEADER ${PROF_API_STR})



# We expect amdhip64_static to contain objects of rocclr and hip. But linker
# let amdhip64_static contain objects of hip only. So we will use a
# a custom amdhip64_static_combiner to combine objects of vid and hip into
# amdhip64_static. To avoid amdhip64_static contains itself,
# amdhip64_static_temp is created internally.
add_library(amdhip64_static_temp STATIC
    $<TARGET_OBJECTS:hip64>
    )

add_library(host INTERFACE)
target_link_libraries(host INTERFACE amdhip64)
add_library(device INTERFACE)
target_link_libraries(device INTERFACE host)
# TODO: we may create host_static and device_static to let app
# link amdhip64_static

# FIXME: Linux convention is to create static library with same base
# filename.
target_link_libraries(amdhip64 PRIVATE amdrocclr_static Threads::Threads dl)
target_link_libraries(amdhip64_static PRIVATE Threads::Threads dl)
target_link_libraries(amdhip64_static_temp PRIVATE Threads::Threads dl)

# combine objects of vid and hip into amdhip64_static
add_custom_target(
    amdhip64_static_combiner
    ALL
    COMMAND rm -f $<TARGET_FILE:amdhip64_static>  # Must remove old one, otherwise the new one will contain obsolete stuff
    COMMAND ${CMAKE_AR} -rcsT $<TARGET_FILE:amdhip64_static> $<TARGET_FILE:amdhip64_static_temp> $<TARGET_FILE:amdrocclr_static>
    DEPENDS amdhip64_static amdhip64_static_temp amdrocclr_static # To make sure this is the last step
    COMMENT "Combining static libs into amdhip64_static"
)


INSTALL(PROGRAMS $<TARGET_FILE:amdhip64_static> DESTINATION lib COMPONENT MAIN)
INSTALL(PROGRAMS $<TARGET_FILE:amdhip64> DESTINATION lib COMPONENT MAIN)
INSTALL(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink libamdhip64.so lib/libhip_hcc.so  )" DESTINATION lib COMPONENT MAIN)

INSTALL(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink libamdhip64.so lib/libhiprtc.so  )" DESTINATION lib COMPONENT MAIN)
INSTALL(FILES ${CMAKE_BINARY_DIR}/lib/libhip_hcc.so DESTINATION lib COMPONENT MAIN)

INSTALL(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc.so DESTINATION lib COMPONENT MAIN)

INSTALL(TARGETS amdhip64 amdhip64_static host device EXPORT hip-targets DESTINATION ${LIB_INSTALL_DIR})
INSTALL(EXPORT hip-targets DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR} NAMESPACE hip::)

