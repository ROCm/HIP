project(static_lib)

cmake_minimum_required(VERSION 3.10)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(hip REQUIRED)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(CPP_SOURCES hipDevice.cpp)

# Generate static lib libHipDevice.a
add_library(HipDevice STATIC ${CPP_SOURCES})

target_compile_options(HipDevice PRIVATE -fgpu-rdc)
target_link_libraries(HipDevice PRIVATE -fgpu-rdc)
target_include_directories(HipDevice PRIVATE /opt/rocm/hsa/include)

# Create test executable that uses libHipDevice.a
set(TEST_SOURCES ${CMAKE_SOURCE_DIR}/hipMain2.cpp)

add_executable(test_device_static ${TEST_SOURCES})
add_dependencies(test_device_static HipDevice)
target_compile_options(test_device_static PRIVATE -fgpu-rdc)
target_link_libraries(test_device_static PRIVATE HipDevice)
target_link_libraries(test_device_static PRIVATE -fgpu-rdc hip::host)

