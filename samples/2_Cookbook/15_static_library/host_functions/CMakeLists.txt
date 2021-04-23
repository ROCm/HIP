project(static_lib)

cmake_minimum_required(VERSION 3.10)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(hip REQUIRED)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_AR           ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

set(CPP_SOURCES hipOptLibrary.cpp)

# Generate static lib libHipOptLibrary.a.
add_library(HipOptLibrary STATIC ${CPP_SOURCES})

# Set-up the correct flags to generate the static library.
target_link_libraries(HipOptLibrary PRIVATE --emit-static-lib)
target_include_directories(HipOptLibrary PRIVATE /opt/rocm/hsa/include)
get_property(link_libraries TARGET HipOptLibrary PROPERTY LINK_LIBRARIES)
string (REPLACE ";" " " LINK_PROPS "${link_libraries}")
set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -o <TARGET> ${LINK_PROPS} <LINK_FLAGS> <OBJECTS>")

# Create test executable that uses libHipOptLibrary.a
set(TEST_SOURCES ${CMAKE_SOURCE_DIR}/hipMain1.cpp)

add_executable(test_opt_static ${TEST_SOURCES})
add_dependencies(test_opt_static HipOptLibrary)
target_link_libraries(test_opt_static PRIVATE -lHipOptLibrary -L${CMAKE_BINARY_DIR})
target_link_libraries(test_opt_static PRIVATE amdhip64 amd_comgr hsa-runtime64::hsa-runtime64)

