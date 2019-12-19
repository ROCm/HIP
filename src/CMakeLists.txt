# Determine HCC_HOME
if(NOT DEFINED HCC_HOME)
    if(NOT DEFINED ENV{HCC_HOME})
        set(HCC_HOME "/opt/rocm/hcc" CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_HOME $ENV{HCC_HOME} CACHE PATH "Path to which HCC has been installed")
    endif()
endif()

if(DEFINED ENV{HIP_DEVELOPER})
    add_to_config(_buildInfo HCC_HOME)
endif()
if(IS_ABSOLUTE ${HCC_HOME} AND EXISTS ${HCC_HOME} AND IS_DIRECTORY ${HCC_HOME})
    execute_process(COMMAND ${HCC_HOME}/bin/hcc --version
        OUTPUT_VARIABLE HCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE ".*based on HCC " "" HCC_VERSION ${HCC_VERSION})
    string(REGEX REPLACE " .*" "" HCC_VERSION ${HCC_VERSION})
    message(STATUS "Looking for HCC in: " ${HCC_HOME} ". Found version: " ${HCC_VERSION})
else()
    message(FATAL_ERROR "Don't know where to find HCC. Please specify abolute path using -DHCC_HOME")
endif()
add_to_config(_buildInfo HCC_VERSION)
string(REPLACE "-" ";" HCC_VERSION_LIST ${HCC_VERSION})
list(GET HCC_VERSION_LIST 0 HCC_PACKAGE_VERSION)
string(REPLACE "." ";" HCC_VERSION_LIST ${HCC_PACKAGE_VERSION})
list(GET HCC_VERSION_LIST 0 HCC_VERSION_MAJOR)
list(GET HCC_VERSION_LIST 1 HCC_VERSION_MINOR)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(HIP_HCC_BUILD_FLAGS)
if(COMPILE_HIP_ATP_MARKER)
    include_directories(/opt/rocm/profiler/CXLActivityLogger/include)
    set(HIP_HCC_BUILD_FLAGS "${HIP_HCC_BUILD_FLAGS} -DCOMPILE_HIP_ATP_MARKER=1")
endif()

# Add HIP_VERSION to CMAKE_<LANG>_FLAGS
set(HIP_HCC_BUILD_FLAGS "${HIP_HCC_BUILD_FLAGS} -DHIP_VERSION_MAJOR=${HIP_VERSION_MAJOR} -DHIP_VERSION_MINOR=${HIP_VERSION_MINOR} -DHIP_VERSION_PATCH=${HIP_VERSION_GITDATE}")

# Add remaining flags
set(HCC_CXX_FLAGS  "-hc -fno-gpu-rdc --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 ")
set(HIP_HCC_BUILD_FLAGS "${HIP_HCC_BUILD_FLAGS} -fPIC ${HCC_CXX_FLAGS} -I${HSA_PATH}/include")

# Set compiler and compiler flags
set(CMAKE_CXX_COMPILER "${HCC_HOME}/bin/hcc")
set(CMAKE_C_COMPILER   "${HCC_HOME}/bin/hcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_BUILD_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HIP_HCC_BUILD_FLAGS}")

set(SOURCE_FILES_RUNTIME
    code_object_bundle.cpp
    program_state.cpp
    hip_clang.cpp
    hip_hcc.cpp
    hip_context.cpp
    hip_device.cpp
    hip_error.cpp
    hip_event.cpp
    hip_fatbin.cpp
    hip_memory.cpp
    hip_peer.cpp
    hip_stream.cpp
    hip_module.cpp
    hip_db.cpp
    grid_launch.cpp
    hip_texture.cpp
    hip_surface.cpp
    hip_intercept.cpp
    env.cpp
    h2f.cpp)

if(COMPILE_HIP_ATP_MARKER)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/opt/rocm/profiler/CXLActivityLogger/bin/x86_64 -lCXLActivityLogger")
endif()

#-------------------------------------hip_hcc----------------------------------#
add_library(hip_hcc SHARED ${SOURCE_FILES_RUNTIME})
add_library(hip_hcc_static STATIC ${SOURCE_FILES_RUNTIME})

set_target_properties(hip_hcc PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(hip_hcc PROPERTIES VISIBILITY_INLINES_HIDDEN 1)

target_link_libraries(hip_hcc PRIVATE hc_am)
target_link_libraries(hip_hcc_static PRIVATE hc_am)

find_package(amd_comgr REQUIRED CONFIG
    PATHS
        /opt/rocm/
    PATH_SUFFIXES
        cmake/amd_comgr
        lib/cmake/amd_comgr
)
MESSAGE(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

target_link_libraries(hip_hcc PRIVATE amd_comgr)
target_link_libraries(hip_hcc_static PRIVATE amd_comgr)

string(REPLACE " " ";" HCC_CXX_FLAGS_LIST ${HCC_CXX_FLAGS})
foreach(TARGET hip_hcc hip_hcc_static)
    target_include_directories(${TARGET} SYSTEM INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>;${HSA_PATH}/include)
endforeach()
#-------------------------------------hip_hcc----------------------------------#

add_library(host INTERFACE)
target_link_libraries(host INTERFACE hip_hcc)

add_library(device INTERFACE)
target_link_libraries(device INTERFACE host hcc::hccrt hcc::hc_am)

# Generate .hipInfo
file(WRITE "${PROJECT_BINARY_DIR}/.hipInfo" ${_buildInfo})

#-------------------------------------hiprtc-----------------------------------#
add_library(hiprtc SHARED hiprtc.cpp code_object_bundle.cpp)
target_include_directories(hiprtc SYSTEM
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${HSA_PATH}/include)
target_link_libraries(hiprtc PUBLIC stdc++fs)
#-------------------------------------hiprtc-----------------------------------#

#############################
# Install steps
#############################
install(TARGETS hip_hcc_static hip_hcc hiprtc DESTINATION lib)
# Install .hipInfo
install(FILES ${PROJECT_BINARY_DIR}/.hipInfo DESTINATION lib)

install(TARGETS hip_hcc_static hip_hcc host device EXPORT hip-targets DESTINATION ${LIB_INSTALL_DIR})
install(EXPORT hip-targets DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR} NAMESPACE hip::)
