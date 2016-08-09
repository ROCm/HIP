cmake_minimum_required (VERSION 2.6)

# remove CMAKE_CXX_COMPILER entry from cache since it will be pointing to hipcc
unset(CMAKE_CXX_COMPILER CACHE)
message (CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER} )

project (HIP_Unit_Tests)
include(CTest)


#include_directories( ${PROJECT_SOURCE_DIR}/include )
set (HIPTEST_SOURCE_DIR ${PROJECT_SOURCE_DIR} )

# The version number.
set (HIP_Unit_Test_VERSION_MAJOR 1)
set (HIP_Unit_Test_VERSION_MINOR 0)

if(NOT DEFINED HIP_MULTI_GPU)
    set(HIP_MULTI_GPU 0 CACHE BOOL "Run tests requiring more than one GPU")
endif()

if(NOT DEFINED HIP_BUILD_LOCAL)
    if(NOT DEFINED ENV{HIP_BUILD_LOCAL})
        set(HIP_BUILD_LOCAL 1 CACHE BOOL "Build HIP in local folder")
    else()
        set(HIP_BUILD_LOCAL $ENV{HIP_BUILD_LOCAL} CACHE BOOL "Build HIP in local folder")
    endif()
endif()

set(HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
    get_filename_component (HIP_PATH ../.. ABSOLUTE)
endif()

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)

MESSAGE ("HIP_PATH=" ${HIP_PATH})

if (${HIP_PLATFORM} STREQUAL "hcc")
	MESSAGE ("HIP_PLATFORM=hcc")

    set (HSA_PATH $ENV{HSA_PATH})
    if (NOT DEFINED HSA_PATH)
        set (HSA_PATH /opt/rocm/hsa)
    endif()

    set (CODEXL_PATH $ENV{CODEXL_PATH})
    if (NOT DEFINED CODEXL_PATH)
        set (CODEXL_PATH /opt/AMD/CodeXL)
    endif()
    set (CODEXL_SDK_ATAL_PATH ${CODEXL_PATH}/SDK/AMDTActivityLogger)

	#---
	# Add HSA library:
	add_library(hsa-runtime64 SHARED IMPORTED)
	set_property(TARGET hsa-runtime64 PROPERTY IMPORTED_LOCATION "${HSA_PATH}/lib/libhsa-runtime64.so") 

	#These includes are used for all files.
	#Include HIP and HC since the tests need both of these:
	include_directories(${HIP_PATH}/include)

    # This will create a subdir "hip_hcc" in the test build directory
    # Any changes to hip_hcc source will be detected and force the library and then the tests to be rebuilt.  
    if (${HIP_BUILD_LOCAL})
        add_subdirectory(${HIP_PATH} build.hip_hcc)
        #link_directories(${CMAKE_CURRENT_BINARY_DIR}/build.hip_hcc)  # search the local hip_hcc for libhip_hcc.a
        set (CMAKE_CXX_FLAGS --hipcc_explicit_lib)
    endif()


elseif (${HIP_PLATFORM} STREQUAL "nvcc")
	MESSAGE ("HIP_PLATFORM=nvcc")

	#Need C++11 for threads in some of the tests.
	add_definitions(-std=c++11)

	# NVCC does not not support -rdynamic option
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS )
	set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS )

else()
	MESSAGE (FATAL_ERROR "UNKNOWN HIP_PLATFORM=" ${HIP_PLATFORM})
endif()


set (HIPCC ${HIP_PATH}/bin/hipcc)
set (CMAKE_CXX_COMPILER ${HIPCC} CACHE FILEPATH "CXX Compiler" FORCE)

add_library(test_common OBJECT test_common.cpp )


# usage : build_hip_executable (exe_name CPP_FILES)
macro (build_hip_executable exe cpp)
    if (${HIP_PLATFORM} STREQUAL "hcc")
        if (${HIP_BUILD_LOCAL})
            #target_link_libraries(${exe} hip_hcc)
            add_executable (${exe} ${cpp} ${ARGN} $<TARGET_OBJECTS:test_common> $<TARGET_OBJECTS:hip_hcc> )
        else()
            add_executable (${exe} ${cpp} ${ARGN} $<TARGET_OBJECTS:test_common> )
        endif()
    else()
        add_executable (${exe} ${cpp} ${ARGN} $<TARGET_OBJECTS:test_common> )
    endif()
endmacro()


# Make a hip executable, using libc++
macro (build_hip_executable_libcpp exe cpp)
    build_hip_executable( ${exe} ${cpp} ${ARGN} )
    if (${HIP_PLATFORM} STREQUAL "hcc")
        set_source_files_properties (${cpp} i${ARGN} PROPERTIES COMPILE_FLAGS --stdlib=libc++ )
    endif()
endmacro()

function (make_named_test exe testname )
  add_test (NAME ${testname}
      COMMAND ${PROJECT_BINARY_DIR}/${exe} ${ARGN}
            )
  set_tests_properties (${testname}
      PROPERTIES PASS_REGULAR_EXPRESSION "PASSED"
    )
endfunction()

macro (make_test exe )
  string (REPLACE " " "" smush_args ${ARGN})
  set (testname ${PROJECT_NAME}/${exe}${smush_args}.tst)

  make_named_test(${exe} ${testname} ${ARGN})
endmacro()


macro (make_hipify_test sourceFile )
    #string (REPLACE " " "" smush_args ${ARGN})
  set (testname ${sourceFile}${smush_args}.tst)

  add_test (NAME ${testname}
      COMMAND ${HIP_PATH}/bin/hipify ${PROJECT_SOURCE_DIR}/${sourceFile} ${ARGN} 
            )
endmacro()


macro (make_test_matches exe match_string)
  string (REPLACE " " "" smush_args ${ARGN})
  set (testname ${exe}${smush_args}.tst)
  add_test (NAME ${testname}
      COMMAND ${PROJECT_BINARY_DIR}/${exe} ${ARGN}
            )
  set_tests_properties (${testname}
      PROPERTIES PASS_REGULAR_EXPRESSION ${match_string}
    )
endmacro()

macro (build_hip_executable_sm35 exe cpp)
  build_hip_executable( ${exe} ${cpp} ${ARGN} )
  if (${HIP_PLATFORM} STREQUAL "nvcc")
    set_source_files_properties (${cpp} i${ARGN} PROPERTIES COMPILE_FLAGS --gpu-architecture=sm_35 ) 
  endif()
endmacro()


build_hip_executable (hipGetDeviceAttribute hipGetDeviceAttribute.cpp) 
build_hip_executable (hipEnvVar hipEnvVar.cpp)
build_hip_executable (hipEnvVarDriver hipEnvVarDriver.cpp) 
build_hip_executable (hipEventRecord hipEventRecord.cpp)

build_hip_executable_libcpp (hipHcc hipHcc.cpp)
#set_source_files_properties (hipHcc.cpp PROPERTIES COMPILE_FLAGS --stdlib=libc++ )

build_hip_executable_libcpp (hipPointerAttrib hipPointerAttrib.cpp)
build_hip_executable (hipHostAlloc hipHostAlloc.cpp)
build_hip_executable (hipHostGetFlags hipHostGetFlags.cpp)
build_hip_executable (hipHostRegister hipHostRegister.cpp)
build_hip_executable (hipRandomMemcpyAsync hipRandomMemcpyAsync.cpp)
build_hip_executable (hipFuncSetDeviceFlags hipFuncSetDeviceFlags.cpp)
build_hip_executable (hipFuncGetDevice hipFuncGetDevice.cpp)
build_hip_executable (hipFuncSetDevice hipFuncSetDevice.cpp)
build_hip_executable (hipFuncDeviceSynchronize hipFuncDeviceSynchronize.cpp)
build_hip_executable (hipPeerToPeer_simple hipPeerToPeer_simple.cpp)
build_hip_executable (hipTestMemcpyPin hipTestMemcpyPin.cpp)
#build_hip_executable (hipDynamicShared hipDynamicShared.cpp)
build_hip_executable (hipLaunchParm hipLaunchParm.cpp)

if (${HIP_PLATFORM} STREQUAL "hcc")
    build_hip_executable (hipArray hipArray.cpp)
endif()

make_test(hipEventRecord --iterations 10)
make_test(hipEnvVarDriver " " )
make_test(hipLaunchParm " ")
#TODO -reenable
#make_test(hipPointerAttrib " " )


make_test(hipHostAlloc " ")
# BS- comment out since test appears broken - asks for device pointer but pointer was never allocated.
#make_test(hipHostGetFlags " ")
make_test(hipHcc  " " )
make_test(hipHostRegister " ")
make_test(hipRandomMemcpyAsync " ")
make_test(hipFuncSetDeviceFlags " ")
make_test(hipFuncGetDevice " ")
make_test(hipFuncDeviceSynchronize " ")
make_test(hipTestMemcpyPin " ")

if (${HIP_MULTI_GPU})
    make_test(hipPeerToPeer_simple " ")                  # use current device for copy, this fails.
    make_test(hipPeerToPeer_simple --memcpyWithPeer)
    make_test(hipPeerToPeer_simple --mirrorPeers)    # mirror mapping: test to ensure mirror doesn't destroy orig mapping.

endif()

if (${HIP_PLATFORM} STREQUAL "hcc")
    make_test(hipArray " ")
    make_test(hipFuncSetDevice " ")
endif()

make_hipify_test(specialFunc.cu )

#make_test(hipDynamicShared " ")

# Add subdirs here:
add_subdirectory(context)
add_subdirectory(deviceLib)
add_subdirectory(runtimeApi)
add_subdirectory(kernel)
