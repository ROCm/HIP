#project("hip")
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-keep-memory -Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all -Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hip_hcc.map.in")

if(CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared-libasan")
endif()

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-D__HIP_VDI__ -D__HIP_PLATFORM_HCC__ -DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER -DUSE_PROF_API)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

#command example:
#cmake -DVDI_DIR=/home/pghafari/vdi/vdi -DLIBOCL_STATIC_DIR=/home/pghafari/git-ocl/build/amdocl -DLIBVDI_STATIC_DIR=/home/pghafari/vdi/vdi/build -DUSE_COMGR_LIBRARY=yes -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/home/pghafari/rocmgitvdi/hsa-runtime/opensrc/hsa-runtime/build;/home/pghafari/rocmgitvdi/hsa-runtime/opensrc;/home/pghafari/rocmgitvdi/support/lib/comgr;/home/pghafari/rocmgitvdi/support/lib/comgr/build;" -DCMAKE_MODULE_PATH=/home/pghafari/rocmgit/opencl/cmake/modules -DLLVM_INCLUDES=/home/pghafari/rocmgit/llvm-project/llvm/include ..

set(USE_PROF_API "1")

find_package(ROCT REQUIRED)
find_package(ROCR REQUIRED)

if(NOT DEFINED LIBVDI_STATIC_DIR)
   find_path(LIBVDI_STATIC_DIR
        NAMES libamdvdi_static.a
        PATHS /opt/rocm/vdi
        PATH_SUFFIXES lib
   )
endif()

if(NOT DEFINED VDI_DIR)
   find_path(VDI_DIR
        NAMES top.hpp
        PATH_SUFFIXES include
        PATHS /opt/rocm/vdi
   )
endif()
message("vdilib:${LIBVDI_STATIC_DIR} ,found vdi includes ${VDI_DIR}")
set(PROF_API_HEADER_PATH ${VDI_DIR}/platform)
#############################
# Profiling API support
#############################
# Generate profiling API macros/structures header
set(PROF_API_STR "${CMAKE_CURRENT_SOURCE_DIR}/../include/hip/hcc_detail/hip_prof_str.h")
set(PROF_API_HDR "${CMAKE_CURRENT_SOURCE_DIR}/../include/hip/hcc_detail/hip_runtime_api.h")
set(PROF_API_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROF_API_GEN "${CMAKE_CURRENT_SOURCE_DIR}/hip_prof_gen.py")
set(PROF_API_LOG "${PROJECT_BINARY_DIR}/hip_prof_gen.log.txt")
set(PROF_API_CMD "${PROF_API_GEN} -v -t --priv ${OPT_PROF_API} ${PROF_API_HDR} ${PROF_API_SRC} ${PROF_API_STR} >${PROF_API_LOG}")
MESSAGE(STATUS "Generating profiling promitives: ${PROF_API_STR}")
execute_process(COMMAND sh -c "rm -f ${PROF_API_STR}; ${PROF_API_CMD}")
MESSAGE(COMMAND sh -c "rm -f ${PROF_API_STR}; ${PROF_API_CMD}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${PROF_API_GEN} ${PROF_API_HDR} ${PROF_API_STR})

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


if(NOT DEFINED VDI_DIR OR NOT DEFINED LIBOCL_STATIC_DIR OR NOT DEFINED LIBVDI_STATIC_DIR )
	#	message(FATAL_ERROR "define VDI_DIR, LIBOCL_STATIC_DIR\n")

endif()
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" )
set(CMAKE_MODULE_PATH${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include_directories(${ROCR_INCLUDES})
if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")
   message(STATUS "LLVM includes found ${LLVM_INCLUDES}")
   include_directories(${LLVM_INCLUDES})
endif() # if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")

#find_package(hsa-runtime REQUIRED CONFIG PATHS ${HSA_RUNTIME} NO_DEFAULT_PATH)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/elfio)
include_directories(${CMAKE_SOURCE_DIR}/amdocl)
include_directories(${CMAKE_SOURCE_DIR}/include/hip/hcc_detail/elfio)
include_directories(${VDI_DIR})
include_directories(${VDI_DIR}/include)
include_directories(${VDI_DIR}/compiler/lib)
include_directories(${VDI_DIR}/compiler/lib/include)
include_directories(${VDI_DIR}/compiler/lib/loaders)
include_directories(${VDI_DIR}/compiler/lib/loaders/elf/utils/common)
include_directories(${VDI_DIR}/compiler/lib/loaders/elf/utils/libelf)
add_definitions(-DUSE_COMGR_LIBRARY -DCOMGR_DYN_DLL)
    find_package(amd_comgr REQUIRED CONFIG
            PATHS
                /opt/rocm/
            PATH_SUFFIXES
                cmake/amd_comgr
                lib/cmake/amd_comgr
    )
    MESSAGE(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

#find_package(amd_comgr REQUIRED CONFIG)
include_directories("$<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>")

add_definitions(-DBSD_LIBELF)
#find_package( Threads REQUIRED CONFIG)
#find_package( LibElf REQUIRED CONFIG)

add_library(amdhip64 SHARED
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
 cl_icd.cpp
 cl_lqdflash_amd.cpp
 )
add_library(amdvdi_static STATIC IMPORTED)
set_target_properties(amdvdi_static PROPERTIES IMPORTED_LOCATION "${LIBVDI_STATIC_DIR}/libamdvdi_static.a")
#add_library(amdocl64 SHARED IMPORTED)
#set_target_properties(amdocl64 PROPERTIES IMPORTED_LOCATION "/opt/rocm/opencl/lib/x86_64/libamdocl64.so")
#add_library(oclelf STATIC IMPORTED)
#set_target_properties(oclelf PROPERTIES IMPORTED_LOCATION "${LIBOCLELF_STATIC_DIR}/liboclelf.a")

    add_library(host INTERFACE)
    target_link_libraries(host INTERFACE amdhip64)
    add_library(device INTERFACE)
    target_link_libraries(device INTERFACE host)
    #target_link_libraries(amdhip64 PRIVATE amd_comgr)

target_link_libraries(amdhip64 amdvdi_static pthread dl ${ROCT_LIBRARIES} ${ROCR_LIBRARIES})
#target_link_libraries(amdhip64 amdvdi_static amdocl64_static pthread dl ${ROCT_LIBRARIES} ${ROCR_LIBRARIES})

#install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink libamdhip64.so lib/libhip_hcc.so)"   )

INSTALL(PROGRAMS $<TARGET_FILE:amdhip64> DESTINATION lib COMPONENT MAIN)
INSTALL(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink libamdhip64.so lib/libhip_hcc.so  )" DESTINATION lib COMPONENT MAIN)

INSTALL(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink libamdhip64.so lib/libhiprtc.so  )" DESTINATION lib COMPONENT MAIN)
INSTALL(FILES ${CMAKE_BINARY_DIR}/lib/libhip_hcc.so DESTINATION lib COMPONENT MAIN)

INSTALL(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc.so DESTINATION lib COMPONENT MAIN)

if(0)
set(CPACK_GENERATOR "DEB;RPM" CACHE STRING "Default packaging generators")

set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_VENDOR "AMD")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm/hip-on-vdi")

set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")

# Debian CPACK variables
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_MAIN_FILE_NAME "hip-on-vdi-1.0.0.deb")
set(CPACK_DEBIAN_MAIN_PACKAGE_NAME "hip-on-vdi")
#set(CPACK_DEBIAN_MAIN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/DEB/postinst;${CMAKE_CURRENT_SOURCE_DIR}/DEB/prerm")

# RPM CPACK variables
set(CPACK_RPM_COMPONENT_INSTALL ON)

set(CPACK_RPM_MAIN_FILE_NAME "hip-on-vdi-1.0.0.rpm")
set(CPACK_RPM_MAIN_PACKAGE_NAME "hip-on-vdi")
#set(CPACK_RPM_MAIN_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
#set(CPACK_RPM_MAIN_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")

set(CPACK_COMPONENTS_ALL MAIN)
include(CPack)
endif()
