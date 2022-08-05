project(static_lib)

cmake_minimum_required(VERSION 3.10)

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# Find hip
find_package(hip REQUIRED)

# For windows, AR is MS Librarian and that is picked by Visual Studio's command prompt.
if (WIN32)
  find_program(libpath NAMES lib.exe)
  set (CMAKE_AR ${libpath})
endif()

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(CPP_SOURCES hipDevice.cpp)

# For windows, We need to tell cmake how to create static library.
if (WIN32)
  set (CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

# Generate static lib libHipDevice.a
add_library(HipDevice STATIC ${CPP_SOURCES})

target_compile_options(HipDevice PRIVATE -fgpu-rdc)
target_link_libraries(HipDevice PRIVATE -fgpu-rdc)
target_include_directories(HipDevice PRIVATE ${ROCM_PATH}/hsa/include)

# Create test executable that uses libHipDevice.a
set(TEST_SOURCES ${CMAKE_SOURCE_DIR}/hipMain2.cpp)

add_executable(test_device_static ${TEST_SOURCES})
add_dependencies(test_device_static HipDevice)
target_compile_options(test_device_static PRIVATE -fgpu-rdc)

# For windows, Change in a way to pass lib details
if (WIN32)
  target_link_libraries(test_device_static PRIVATE -lHipDevice -L${CMAKE_BINARY_DIR})
else()
  target_link_libraries(test_device_static PRIVATE HipDevice)
endif()

target_link_libraries(test_device_static PRIVATE -fgpu-rdc amdhip64 amd_comgr)

