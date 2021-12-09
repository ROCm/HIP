cmake_minimum_required(VERSION 3.16.8)
project(hiptests)

# Check if platform and compiler are set
if(HIP_PLATFORM STREQUAL "amd")
    if(HIP_COMPILER STREQUAL "nvcc")
        message(FATAL_ERROR "Unexpected HIP_COMPILER:${HIP_COMPILER} is set for HIP_PLATFOR:amd")
    endif()
elseif(HIP_PLATFORM STREQUAL "nvidia")
    if(NOT DEFINED HIP_COMPILER OR NOT HIP_COMPILER STREQUAL "nvcc")
        message(FATAL_ERROR "Unexpected HIP_COMPILER: ${HIP_COMPILER} is set for HIP_PLATFORM:nvidia")
    endif()
else()
    message(FATAL_ERROR "Unexpected HIP_PLATFORM: " ${HIP_PLATFORM})
endif()

# Set HIP Path
if(NOT DEFINED HIP_PATH)
    if(DEFINED ENV{HIP_PATH})
        set(HIP_PATH $ENV{HIP_PATH} CACHE STRING "HIP Path")
    else()
        set(HIP_PATH "${PROJECT_BINARY_DIR}")
    endif()
endif()
message(STATUS "HIP Path: ${HIP_PATH}")

if(UNIX)
    set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")
    set(CMAKE_C_COMPILER "${HIP_PATH}/bin/hipcc")
else()
    # using cmake_path as it handles path correctly.
    # Set both compilers else windows cmake complains of mismatch
    cmake_path(SET CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc.bat")
    cmake_path(SET CMAKE_C_COMPILER "${HIP_PATH}/bin/hipcc.bat")
endif()


if(NOT UNIX)
    # In linux this reruns the cmake and fails with incorrect vars.
    # so the project command is used only for windows
    project(build_tests)
endif()

if(NOT DEFINED CATCH2_PATH)
    if(DEFINED ENV{CATCH2_PATH})
        set(CATCH2_PATH $ENV{CATCH2_PATH} CACHE STRING "Catch2 Path")
    else()
        set(CATCH2_PATH "${CMAKE_CURRENT_LIST_DIR}/external/Catch2")
    endif()
endif()
message(STATUS "Catch2 Path: ${CATCH2_PATH}")

# Set JSON Parser path
if(NOT DEFINED JSON_PARSER)
    if(DEFINED ENV{JSON_PARSER})
        set(JSON_PARSER $ENV{JSON_PARSER} CACHE STRING "JSON Parser Path")
    else()
        set(JSON_PARSER "${CMAKE_CURRENT_LIST_DIR}/external/picojson")
    endif()
endif()

message(STATUS "Searching Catch2 in: ${CMAKE_CURRENT_LIST_DIR}/external")
find_package(Catch2 REQUIRED
    PATHS
        ${CMAKE_CURRENT_LIST_DIR}/external
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

# Turn off CMAKE_HIP_ARCHITECTURES Feature if cmake version is 3.21+
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
    set(CMAKE_HIP_ARCHITECTURES OFF)
endif()
message(STATUS "CMAKE HIP ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

# Identify the GPU Targets.
# This is done due to limitation of rocm_agent_enumerator
# While building test parallelly, rocm_agent_enumerator can fail and give out an empty target
# That results in hipcc building the test for gfx803 (the default target)
if(NOT DEFINED OFFLOAD_ARCH_STR AND EXISTS "${ROCM_PATH}/bin/rocm_agent_enumerator"
   AND HIP_PLATFORM STREQUAL "amd" AND UNIX)
    execute_process(COMMAND ${ROCM_PATH}/bin/rocm_agent_enumerator OUTPUT_VARIABLE HIP_GPU_ARCH
         RESULT_VARIABLE ROCM_AGENT_ENUM_RESULT)
    # Trim out gfx000
    string(REPLACE "gfx000\n" "" HIP_GPU_ARCH ${HIP_GPU_ARCH})
    string(LENGTH ${HIP_GPU_ARCH} HIP_GPU_ARCH_LEN)

    # If string has more gfx target except gfx000
    if(${HIP_GPU_ARCH_LEN} GREATER_EQUAL 1)
		string(REGEX REPLACE "\n" ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH}")
		set(OFFLOAD_ARCH_STR "")
		foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
			set(OFFLOAD_ARCH_STR " ${OFFLOAD_ARCH_STR} --offload-arch=${_hip_gpu_arch} ")
		endforeach()
		message(STATUS "Using offload arch string: ${OFFLOAD_ARCH_STR}")
    endif()
endif()

if(DEFINED OFFLOAD_ARCH_STR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OFFLOAD_ARCH_STR} ")
endif()

# Disable CXX extensions (gnu++11 etc)
set(CMAKE_CXX_EXTENSIONS OFF)

add_custom_target(build_tests)

# Tests folder
add_subdirectory(unit)
add_subdirectory(ABM)
add_subdirectory(hipTestMain)
add_subdirectory(stress)

if(UNIX)
    add_subdirectory(multiproc)
    add_subdirectory(TypeQualifiers)
endif()

cmake_policy(POP)
