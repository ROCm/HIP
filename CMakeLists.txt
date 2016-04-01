cmake_minimum_required(VERSION 2.6)
project(hip_hcc)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED HCC_DIR)
    if(NOT DEFINED ENV{HCC_DIR})
        set(HCC_DIR "/opt/hcc" CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_DIR $ENV{HCC_DIR} CACHE PATH "Path to which HCC has been installed")
    endif()
endif()
if(IS_ABSOLUTE ${HCC_DIR} AND EXISTS ${HCC_DIR} AND IS_DIRECTORY ${HCC_DIR})
    message(STATUS "Looking for HCC in: " ${HCC_DIR})
else()
    message(FATAL_ERROR "Don't know where to find HCC. Please specify abolute path using -DHCC_DIR")
endif()

if(NOT DEFINED HSA_DIR)
    if(NOT DEFINED ENV{HSA_DIR})
        set(HSA_DIR "/opt/hsa" CACHE PATH "Path to which HSA runtime has been installed")
    else()
        set(HSA_DIR $ENV{HSA_DIR} CACHE PATH "Path to which HSA runtime has been installed")
    endif()
endif()
if(IS_ABSOLUTE ${HSA_DIR} AND EXISTS ${HSA_DIR} AND IS_DIRECTORY ${HSA_DIR})
    message(STATUS "Looking for HSA runtime in: " ${HSA_DIR})
else()
    message(FATAL_ERROR "Don't know where to find HSA runtime. Please specify absolute path using -DHSA_DIR")
endif()

if(NOT DEFINED HIP_INSTALL_DIR)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(HIP_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Installation path for HIP")
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        set(HIP_INSTALL_DIR "/opt/hip" CACHE PATH "Installation path for HIP")
    else()
        message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE specified. Valid values are Debug and Release")
    endif()
endif()
if(IS_ABSOLUTE ${HIP_INSTALL_DIR})
    message(STATUS "HIP will be installed in: " ${HIP_INSTALL_DIR})
else()
    message(FATAL_ERROR "Don't know where to install HIP. Please specify absolute path using -DHIP_INSTALL_DIR")
endif()

set(CMAKE_INSTALL_PREFIX "${HIP_INSTALL_DIR}" CACHE INTERNAL "Installation path for HIP" FORCE)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_COMPILER "${HCC_DIR}/bin/hcc")
set(CMAKE_C_COMPILER   "${HCC_DIR}/bin/hcc")

set(CMAKE_CXX_FLAGS " -hc -I${HCC_DIR}/include -I${HSA_DIR}/include -stdlib=libc++ ")
set(CMAKE_C_FLAGS   " -hc -I${HCC_DIR}/include -I${HSA_DIR}/include -stdlib=libc++ ")

set(SOURCE_FILES src/device_util.cpp
src/hip_hcc.cpp 
src/hip_device.cpp
src/hip_error.cpp
src/hip_event.cpp
src/hip_memory.cpp
src/hip_peer.cpp
src/hip_stream.cpp
src/staging_buffer.cpp)

#add_library(hip_hcc STATIC ${SOURCE_FILES})
add_library(hip_hcc OBJECT ${SOURCE_FILES})

#install(TARGETS hip_hcc DESTINATION ${HIP_INSTALL_DIR}/lib)
install(DIRECTORY ${PROJECT_BINARY_DIR}/CMakeFiles/hip_hcc.dir/src/ DESTINATION ${HIP_INSTALL_DIR}/lib)
set(EXECUTE_COMMAND test ${HIP_INSTALL_DIR} -ef ${CMAKE_CURRENT_SOURCE_DIR})
execute_process(COMMAND ${EXECUTE_COMMAND} RESULT_VARIABLE INSTALL_SOURCE)
if(NOT ${INSTALL_SOURCE} EQUAL 0)
    install(DIRECTORY src DESTINATION ${HIP_INSTALL_DIR})
    install(DIRECTORY bin DESTINATION ${HIP_INSTALL_DIR} USE_SOURCE_PERMISSIONS)
    install(DIRECTORY include DESTINATION ${HIP_INSTALL_DIR})
endif()

