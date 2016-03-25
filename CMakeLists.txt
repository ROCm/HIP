cmake_minimum_required(VERSION 2.6)
project(hip_hcc)

# preserve HCC_HOME env in the generated

if (NOT DEFINED HCC_DIR)
    set (HCC_DIR "/opt/hcc")
endif()
message(STATUS ${HCC_DIR})

if(NOT DEFINED HIP_INSTALL_DIR)
    set(HIP_INSTALL_DIR "/opt/hip")
endif()
message(STATUS ${HIP_INSTALL_DIR})

if(NOT DEFINED HSA_DIR)
    set(HSA_DIR "/opt/hsa")
endif()
message(STATUS ${HSA_DIR})


#if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#    set(CMAKE_INSTALL_PREFIX "${HIP_INSTALL_PATH}" CACHE PATH "Default installation path of hip" FORCE)
#endif ()

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


