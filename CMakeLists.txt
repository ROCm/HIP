cmake_minimum_required(VERSION 2.6)
project(hip_hcc)

set (HCC_DIR "/opt/hcc" CACHE PATH "Path to which HCC has been installed")
message(STATUS "Looking for HCC in: " ${HCC_DIR})

set(HSA_DIR "/opt/hsa" CACHE PATH "Path to which HSA runtime has been installed")
message(STATUS "Looking for HSA runtime in: " ${HSA_DIR})

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(HIP_INSTALL_DIR ${CMAKE_SOURCE_DIR} CACHE PATH "Installation path for HIP")
else()
    set(HIP_INSTALL_DIR "/opt/hip" CACHE PATH "Installation path for HIP")
endif()
message(STATUS "HIP will be installed in: " ${HIP_INSTALL_DIR})

set(CMAKE_INSTALL_PREFIX "${HIP_INSTALL_PATH}" CACHE INTERNAL "Installation path for HIP" FORCE)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_COMPILER "${HCC_DIR}/bin/hcc")
set(CMAKE_C_COMPILER   "${HCC_DIR}/bin/hcc")

set(CMAKE_CXX_FLAGS " -hc -I${HCC_DIR}/include -I${HSA_DIR}/include -stdlib=libc++ ")
set(CMAKE_C_FLAGS   " -hc -I${HCC_DIR}/include -I${HSA_DIR}/include -stdlib=libc++ ")

set(SOURCE_FILES src/hip_hcc.cpp 
src/hip_device.cpp
src/hip_error.cpp
src/hip_event.cpp
src/hip_memory.cpp
src/hip_peer.cpp
src/hip_stream.cpp
src/staging_buffer.cpp)

add_library(hip_hcc STATIC ${SOURCE_FILES})

install(TARGETS hip_hcc DESTINATION ${HIP_INSTALL_DIR}/lib)
install(DIRECTORY src DESTINATION ${HIP_INSTALL_DIR})
install(DIRECTORY bin DESTINATION ${HIP_INSTALL_DIR}  USE_SOURCE_PERMISSIONS)
install(DIRECTORY include DESTINATION ${HIP_INSTALL_DIR})


