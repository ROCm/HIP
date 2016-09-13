cmake_minimum_required(VERSION 2.6)

# remove CMAKE_CXX_COMPILER entry from cache since it will be pointing to hipcc
unset(CMAKE_CXX_COMPILER CACHE)
# remove HIP_PATH entry from cache since we might be running tests with a different configuration
unset(HIP_PATH CACHE)

project(HIP_Unit_Tests)
include(CTest)
set(HIPTEST_SOURCE_DIR ${PROJECT_SOURCE_DIR})

string(ASCII 27 Esc)
set(ColorReset "${Esc}[m")
set(Red         "${Esc}[31m")
set(Magenta     "${Esc}[35m")

# Enable multi-gpu tests
if(NOT DEFINED HIP_MULTI_GPU)
    set(HIP_MULTI_GPU 0 CACHE BOOL "Run tests requiring more than one GPU")
endif()

# We are going to use HIP source...
get_filename_component(HIP_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# ...so we first need to determine the options to cascade to HIP build.
if(DEFINED HIP_PLATFORM)
    set(ENV{HIP_PLATFORM} ${HIP_PLATFORM})
endif()
if(DEFINED HCC_HOME)
    get_filename_component(HCC_HOME ${HCC_HOME} ABSOLUTE)
    set(ENV{HCC_HOME} ${HCC_HOME})
    set(ENV{HIP_DEVELOPER} 1)
endif()
if(DEFINED HIP_LIB_TYPE)
    set(ENV{HIP_LIB_TYPE} ${HIP_LIB_TYPE})
endif()

# Purge previous HIP installation...
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E remove_directory hip
    OUTPUT_QUIET
    ERROR_QUIET
    )
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E make_directory hip
    OUTPUT_QUIET
    ERROR_QUIET
    )
message(STATUS "Configuring HIP")

# ...and now build HIP locally.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/hip/localbuild ${HIP_SRC_PATH}
    WORKING_DIRECTORY hip
    RESULT_VARIABLE hip_build_result
    OUTPUT_VARIABLE hip_build_log
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    )
message("${Magenta}${hip_build_log}${ColorReset}")

if(hip_build_result)
    message(FATAL_ERROR "Error configuring HIP")
else()
    message(STATUS "Configuring HIP - done")
    message(STATUS "Building HIP")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build . --target install
    WORKING_DIRECTORY hip
    RESULT_VARIABLE hip_build_result
    OUTPUT_VARIABLE hip_build_log
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    )

# Show HIP build errors if any.
if(hip_build_result)
    message("${Red}${hip_build_log}${ColorReset}")
    message(FATAL_ERROR "Error building HIP")
else()
    # Building HIP is successful. Point HIP_PATH to this location.
    message(STATUS "Building HIP - done")
    get_filename_component(HIP_PATH ${CMAKE_CURRENT_BINARY_DIR}/hip/localbuild ABSOLUTE)
endif()

# Add a target to rebuild HIP if HIP source changes.
add_custom_target(
    hip ALL
    COMMAND "${CMAKE_COMMAND}" --build . --target install
    WORKING_DIRECTORY hip
    )

# Determine HIP_PLATFORM
execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)

if(${HIP_PLATFORM} STREQUAL "hcc")
    MESSAGE("HIP_PLATFORM=hcc")
elseif(${HIP_PLATFORM} STREQUAL "nvcc")
    MESSAGE("HIP_PLATFORM=nvcc")

    #Need C++11 for threads in some of the tests.
    add_definitions(-std=c++11)

    # NVCC does not not support -rdynamic option
    set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
    set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
else()
    MESSAGE(FATAL_ERROR "UNKNOWN HIP_PLATFORM=" ${HIP_PLATFORM})
endif()

set(HIPCC ${HIP_PATH}/bin/hipcc)
set(CMAKE_CXX_COMPILER ${HIPCC} CACHE FILEPATH "CXX Compiler" FORCE)

add_library(test_common OBJECT test_common.cpp)

# usage : build_hip_executable(exe_name CPP_FILES)
macro(build_hip_executable exe cpp)
    add_executable(${exe} ${cpp} ${ARGN} $<TARGET_OBJECTS:test_common>)
    add_dependencies(${exe} hip)
endmacro()

# Make a hip executable, using libc++
macro(build_hip_executable_libcpp exe cpp)
    build_hip_executable( ${exe} ${cpp} ${ARGN})
    if(${HIP_PLATFORM} STREQUAL "hcc")
        set_source_files_properties(${cpp} i${ARGN} PROPERTIES COMPILE_FLAGS --stdlib=libc++)
    endif()
endmacro()

function(make_named_test exe testname)
    add_test(NAME ${testname}
        COMMAND ${PROJECT_BINARY_DIR}/${exe} ${ARGN}
        )
    set_tests_properties(${testname}
        PROPERTIES PASS_REGULAR_EXPRESSION "PASSED"
        )
endfunction()

macro(make_test exe)
    string(REPLACE " " "" smush_args ${ARGN})
    set(testname ${PROJECT_NAME}/${exe}${smush_args}.tst)

    make_named_test(${exe} ${testname} ${ARGN})
endmacro()


macro(make_hipify_test sourceFile)
    #string(REPLACE " " "" smush_args ${ARGN})
    set(testname ${sourceFile}${smush_args}.tst)

    add_test(NAME ${testname}
        COMMAND ${HIP_PATH}/bin/hipify ${PROJECT_SOURCE_DIR}/${sourceFile} ${ARGN}
        )
endmacro()


macro(make_test_matches exe match_string)
    string(REPLACE " " "" smush_args ${ARGN})
    set(testname ${exe}${smush_args}.tst)
    add_test(NAME ${testname}
        COMMAND ${PROJECT_BINARY_DIR}/${exe} ${ARGN}
        )
    set_tests_properties(${testname}
        PROPERTIES PASS_REGULAR_EXPRESSION ${match_string}
        )
endmacro()

macro(build_hip_executable_sm35 exe cpp)
    build_hip_executable( ${exe} ${cpp} ${ARGN})
    if(${HIP_PLATFORM} STREQUAL "nvcc")
        set_source_files_properties(${cpp} i${ARGN} PROPERTIES COMPILE_FLAGS --gpu-architecture=sm_35)
    endif()
endmacro()


build_hip_executable(hipGetDeviceAttribute hipGetDeviceAttribute.cpp)
build_hip_executable(hipEnvVar hipEnvVar.cpp)
build_hip_executable(hipEnvVarDriver hipEnvVarDriver.cpp)
build_hip_executable(hipEventRecord hipEventRecord.cpp)

build_hip_executable_libcpp(hipHcc hipHcc.cpp)
#set_source_files_properties(hipHcc.cpp PROPERTIES COMPILE_FLAGS --stdlib=libc++)

# __workweek fix.
#build_hip_executable_libcpp(hipPointerAttrib hipPointerAttrib.cpp)
build_hip_executable(hipHostAlloc hipHostAlloc.cpp)
build_hip_executable(hipHostGetFlags hipHostGetFlags.cpp)
build_hip_executable(hipHostRegister hipHostRegister.cpp)
build_hip_executable(hipRandomMemcpyAsync hipRandomMemcpyAsync.cpp)
build_hip_executable(hipFuncSetDeviceFlags hipFuncSetDeviceFlags.cpp)
build_hip_executable(hipFuncGetDevice hipFuncGetDevice.cpp)
build_hip_executable(hipFuncSetDevice hipFuncSetDevice.cpp)
build_hip_executable(hipFuncDeviceSynchronize hipFuncDeviceSynchronize.cpp)
build_hip_executable(hipPeerToPeer_simple hipPeerToPeer_simple.cpp)
build_hip_executable(hipTestMemcpyPin hipTestMemcpyPin.cpp)
build_hip_executable(hipDynamicShared hipDynamicShared.cpp)
build_hip_executable(hipLaunchParm hipLaunchParm.cpp)

if(${HIP_PLATFORM} STREQUAL "hcc")
    build_hip_executable(hipArray hipArray.cpp)
endif()

make_test(hipEventRecord --iterations 10)
make_test(hipEnvVarDriver " ")
make_test(hipLaunchParm " ")
#TODO -reenable
#make_test(hipPointerAttrib " ")


make_test(hipHostAlloc " ")
# BS- comment out since test appears broken - asks for device pointer but pointer was never allocated.
#make_test(hipHostGetFlags " ")
make_test(hipHcc  " ")
make_test(hipHostRegister " ")
make_test(hipRandomMemcpyAsync " ")
make_test(hipFuncSetDeviceFlags " ")
make_test(hipFuncGetDevice " ")
make_test(hipFuncDeviceSynchronize " ")
make_test(hipTestMemcpyPin " ")

if(${HIP_MULTI_GPU})
    make_test(hipPeerToPeer_simple " ")                  # use current device for copy, this fails.
    make_test(hipPeerToPeer_simple --memcpyWithPeer)
    make_test(hipPeerToPeer_simple --mirrorPeers)    # mirror mapping: test to ensure mirror doesn't destroy orig mapping.

endif()

if(${HIP_PLATFORM} STREQUAL "hcc")
    make_test(hipArray " ")
    make_test(hipFuncSetDevice " ")
    make_test(hipDynamicShared " ")
endif()

make_hipify_test(specialFunc.cu)


# Add subdirs here:
add_subdirectory(context)
add_subdirectory(deviceLib)
add_subdirectory(runtimeApi)
add_subdirectory(kernel)
# vim: ts=4:sw=4:expandtab:smartindent
