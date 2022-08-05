project(static_lib)

cmake_minimum_required(VERSION 3.10)

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# Find hip
find_package(hip REQUIRED)

# For windows, AR is MS Librarian and that is pickedby Visual Studio's command prompt.
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

set(CPP_SOURCES hipOptLibrary.cpp)

# For windows, We need to tell cmake how to create static library.
if (WIN32)
  set (CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

# Generate static lib libHipOptLibrary.a.
add_library(HipOptLibrary STATIC ${CPP_SOURCES})

# Set-up the correct flags to generate the static library.
target_link_libraries(HipOptLibrary PRIVATE --emit-static-lib)
target_include_directories(HipOptLibrary PRIVATE /opt/rocm/hsa/include)

# Create test executable that uses libHipOptLibrary.a
set(TEST_SOURCES ${CMAKE_SOURCE_DIR}/hipMain1.cpp)

add_executable(test_opt_static ${TEST_SOURCES})
add_dependencies(test_opt_static HipOptLibrary)
target_link_libraries(test_opt_static PRIVATE -lHipOptLibrary -L${CMAKE_BINARY_DIR})

if (WIN32)
  target_link_libraries(test_opt_static PRIVATE amdhip64 amd_comgr)
else()
  target_link_libraries(test_opt_static PRIVATE amdhip64 amd_comgr hsa-runtime64::hsa-runtime64)
endif()

