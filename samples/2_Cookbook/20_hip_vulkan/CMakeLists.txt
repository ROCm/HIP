
# Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# hipcc.bat fails to qualify as a valid compiler for CMAKE_CXX_COMPILER_ID = ROCMClang
# so  the simple compiler test is skipped and forced to use hipcc.bat as compiler
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
set(CMAKE_CXX_STANDARD 14)
project(hipVulkan)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# need to set rocm_path for windows
# since clang and hip are two different folders during build/install step
if (WIN32 AND HIPINFO_INTERNAL_BUILD)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --rocm-path=${HIP_PATH}")
endif()



# Find hip
find_package(hip REQUIRED)
if (WIN32)
find_package(GLFW3)
if(NOT GLFW_FOUND)
  if(EXISTS "${GLFW_PATH}")
    message(STATUS "FOUND GLFW SDK: ${GLFW_PATH}")
  elseif (EXISTS "$ENV{GLFW_PATH}")
    message(STATUS "FOUND GLFW SDK: $ENV{GLFW_PATH}")
    set(GLFW_PATH  $ENV{GLFW_PATH})
  else()
    message("Error: Unable to locate GLFW SDK. please specify GLFW_PATH")
    return()
  endif()
endif()
endif(WIN32)
find_package(Vulkan)
if(NOT Vulkan_FOUND)
  if(EXISTS "${VULKAN_PATH}")
    message(STATUS "Vulkan SDK: ${VULKAN_PATH}")
  elseif (EXISTS "$ENV{VULKAN_SDK}")
    message(STATUS "FOUND VULKAN SDK: $ENV{VULKAN_SDK}")
    set(VULKAN_PATH  $ENV{VULKAN_SDK})
  else()
    message("Error: Unable to locate Vulkan SDK. please specify VULKAN_PATH")
    return()
  endif()
endif()
set(VULKAN_PATH ${Vulkan_INCLUDE_DIRS})
STRING(REGEX REPLACE "/[Ii]nclude" "" VULKAN_PATH ${VULKAN_PATH})
# Include Vulkan header files from Vulkan SDK
include_directories(AFTER ${VULKAN_PATH}/include)
link_directories(${VULKAN_PATH}/bin;${VULKAN_PATH}/lib;)
link_directories(${GLFW_PATH}/lib-vc2019)
# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Create the excutable
add_executable(hipVulkan VulkanBaseApp.cpp VulkanBaseApp.h main.cpp SineWaveSimulation.cpp SineWaveSimulation.h linmath.h)
include_directories(${HIP_PATH}/include)
include_directories(${GLFW_PATH}/include)

# Link with HIP
if (WIN32)
  target_link_libraries(hipVulkan advapi32 hip::host vulkan-1 glfw3dll)
else (WIN32)
  target_link_libraries(hipVulkan hip::host vulkan glfw)
endif (WIN32)

