cmake_minimum_required(VERSION 2.6)
project(hip_hcc)

set(HCC_PATH "/opt/hcc")
set(HSA_PATH "/opt/hsa")

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_COMPILER "hcc")
set(CMAKE_C_COMPILER "hcc")

set(CMAKE_C_FLAGS " -hc -I${HCC_PATH}/include -I${HSA_PATH}/include -I${HIP_PATH}/include -stdlib=libc++ ")
set(CMAKE_CXX_FLAGS " -hc -I${HCC_PATH}/include -I${HSA_PATH}/include -I${HIP_PATH}/include -stdlib=libc++ ")

set(SOURCE_FILES src/hip_hcc.cpp src/hip_device.cpp src/staging_buffer.cpp)

add_library(hip_hcc STATIC ${SOURCE_FILES})

install(TARGETS hip_hcc DESTINATION /opt/hip/lib)
install(DIRECTORY src DESTINATION /opt/hip/src)
install(DIRECTORY bin DESTINATION /opt/hip/bin)
install(DIRECTORY include DESTINATION /opt/hip/include)


