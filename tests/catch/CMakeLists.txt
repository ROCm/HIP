# Set HIP Path
if(NOT DEFINED HIP_PATH)
    if(DEFINED ENV{HIP_PATH})
        set(HIP_PATH $ENV{HIP_PATH} CACHE STRING "HIP Path")
    else()
        set(HIP_PATH "${PROJECT_BINARY_DIR}")
    endif()
endif()
message(STATUS "HIP Path: ${HIP_PATH}")

set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")

if(NOT DEFINED CATCH2_PATH)
    if(DEFINED ENV{CATCH2_PATH})
        set(CATCH2_PATH $ENV{CATCH2_PATH} CACHE STRING "Catch2 Path")
    else()
        set(CATCH2_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2")
    endif()
endif()
message(STATUS "Catch2 Path: ${CATCH2_PATH}")

# Set JSON Parser path
if(NOT DEFINED JSON_PARSER)
    if(DEFINED ENV{JSON_PARSER})
        set(JSON_PARSER $ENV{JSON_PARSER} CACHE STRING "JSON Parser Path")
    else()
        set(JSON_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/external/picojson")
    endif()
endif()

message(STATUS "Searching Catch2 in: ${CMAKE_CURRENT_SOURCE_DIR}/external")
find_package(Catch2 REQUIRED
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/external
    PATH_SUFFIXES
    Catch2/cmake/Catch2
)
include(Catch)
include(CTest)

include_directories(
    ${CATCH2_PATH}
    "./include"
    ${HIP_PATH}/include
    ${JSON_PARSER}
)

if(HIP_PLATFORM MATCHES "amd" AND HIP_COMPILER MATCHES "clang")
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

cmake_policy(PUSH)
if(POLICY CMP0037)
    cmake_policy(SET CMP0037 OLD)
endif()

# Use clang as host compiler with nvcc
if(HIP_COMPILER MATCHES "nvcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ccbin clang")
endif()

add_custom_target(build_tests)

add_subdirectory(unit)
add_subdirectory(hipTestMain)
add_subdirectory(stress)
add_subdirectory(multiproc)
cmake_policy(POP)
