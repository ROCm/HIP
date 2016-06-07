cmake_minimum_required (VERSION 2.6)
project (HIP_Unit_Tests)
include(CTest)

include_directories( ${PROJECT_SOURCE_DIR}/include )

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
	set (HIP_PATH ../..)
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
set (CMAKE_CXX_COMPILER ${HIPCC})


add_library(test_common OBJECT test_common.cpp )


# usage : make_hip_executable (exe_name CPP_FILES)
macro (make_hip_executable exe cpp)  
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
macro (make_hip_executable_libcpp exe cpp)  
    make_hip_executable( ${exe} ${cpp} ${ARGN} )
    if (${HIP_PLATFORM} STREQUAL "hcc")
        set_source_files_properties (${cpp} i${ARGN} PROPERTIES COMPILE_FLAGS --stdlib=libc++ )
    endif()
endmacro()

macro (make_named_test exe testname )
  add_test (NAME ${testname} 
      COMMAND ${PROJECT_BINARY_DIR}/${exe} ${ARGN}
            )
  set_tests_properties (${testname}
      PROPERTIES PASS_REGULAR_EXPRESSION "PASSED"
    )
endmacro()

macro (make_test exe )
  string (REPLACE " " "" smush_args ${ARGN})
  set (testname ${exe}${smush_args}.tst)

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



#make_hip_executable (hipAPIStreamEnable hipAPIStreamEnable.cpp)
#make_hip_executable (hipAPIStreamDisable hipAPIStreamDisable.cpp)
make_hip_executable (hip_ballot hip_ballot.cpp)
make_hip_executable (hip_anyall hip_anyall.cpp) 
make_hip_executable (hip_popc hip_popc.cpp) 
make_hip_executable (hip_clz hip_clz.cpp) 
make_hip_executable (hip_brev hip_brev.cpp) 
make_hip_executable (hip_ffs hip_ffs.cpp) 
if (${HIP_PLATFORM} STREQUAL "hcc")
    make_hip_executable (hip_test_ldg hip_test_ldg.cpp)
endif()
make_hip_executable (hipGetDeviceAttribute hipGetDeviceAttribute.cpp) 
make_hip_executable (hipEnvVar hipEnvVar.cpp) 
make_hip_executable (hipEnvVarDriver hipEnvVarDriver.cpp) 
make_hip_executable (hipMemcpy_simple hipMemcpy_simple.cpp) 
make_hip_executable (hipMemcpy hipMemcpy.cpp) 
make_hip_executable (hipMemcpyAsync hipMemcpyAsync.cpp) 
make_hip_executable (hipMemset hipMemset.cpp) 
make_hip_executable (hipEventRecord hipEventRecord.cpp) 
make_hip_executable_libcpp (hipLanguageExtensions hipLanguageExtensions.cpp) 
make_hip_executable (hipGridLaunch hipGridLaunch.cpp) 

make_hip_executable_libcpp (hipHcc hipHcc.cpp)
#set_source_files_properties (hipHcc.cpp PROPERTIES COMPILE_FLAGS --stdlib=libc++ ) 

make_hip_executable (hipSimpleAtomicsTest hipSimpleAtomicsTest.cpp)
make_hip_executable (hipMathFunctionsHost hipMathFunctions.cpp hipSinglePrecisionMathHost.cpp hipDoublePrecisionMathHost.cpp)
make_hip_executable (hipMathFunctionsDevice hipMathFunctions.cpp hipSinglePrecisionMathDevice.cpp hipDoublePrecisionMathDevice.cpp)
make_hip_executable (hipIntrinsics hipMathFunctions.cpp hipSinglePrecisionIntrinsics.cpp hipDoublePrecisionIntrinsics.cpp hipIntegerIntrinsics.cpp)
make_hip_executable_libcpp (hipPointerAttrib hipPointerAttrib.cpp) 
make_hip_executable (hipMultiThreadStreams1 hipMultiThreadStreams1.cpp)
make_hip_executable (hipMultiThreadStreams2 hipMultiThreadStreams2.cpp) 
make_hip_executable (hipHostAlloc hipHostAlloc.cpp)
make_hip_executable (hipStreamL5 hipStreamL5.cpp)
make_hip_executable (hipHostGetFlags hipHostGetFlags.cpp)
make_hip_executable (hipHostRegister hipHostRegister.cpp)
make_hip_executable (hipRandomMemcpyAsync hipRandomMemcpyAsync.cpp)
make_hip_executable (hipMemoryAllocate hipMemoryAllocate.cpp)
make_hip_executable (hipFuncSetDeviceFlags hipFuncSetDeviceFlags.cpp)
make_hip_executable (hipFuncGetDevice hipFuncGetDevice.cpp)
make_hip_executable (hipFuncSetDevice hipFuncSetDevice.cpp)
make_hip_executable (hipFuncDeviceSynchronize hipFuncDeviceSynchronize.cpp)
make_hip_executable (hipPeerToPeer_simple hipPeerToPeer_simple.cpp) 
make_hip_executable (hipMemcpyAll hipMemcpyAll.cpp)
make_hip_executable (hipMultiThreadDevice hipMultiThreadDevice.cpp) 
make_hip_executable (hipTestMemcpyPin hipTestMemcpyPin.cpp)
make_hip_executable (hipDynamicShared hipDynamicShared.cpp)

make_test(hip_ballot " " )
make_test(hip_anyall " " )
make_test(hip_popc " " )
make_test(hip_brev " " )
make_test(hip_clz " " )
make_test(hip_ffs " " )
make_test(hipEventRecord --iterations 10)
make_test(hipMemset " " )
make_test(hipMemset --N 10    --memsetval 0x42 )  # small copy, just 10 bytes.
make_test(hipMemset --N 10013 --memsetval 0x5a )  # oddball size.
make_test(hipMemset --N 256M  --memsetval 0xa6 )  # big copy
make_test(hipGridLaunch " " )
make_test(hipEnvVarDriver " " )
make_test(hipMemcpyAll " ")
#TODO -reenable
#make_test(hipPointerAttrib " " )
#make_test(hipMultiThreadStreams1 " " )  Fails if 0x3 specified, passes otherwise.  

make_test(hipMultiThreadStreams2 " " )
make_test(hipMemcpy_simple  " " )
make_named_test(hipMemcpy "hipMemcpy-modes" --tests 0x1 )
make_named_test(hipMemcpy "hipMemcpy-size" --tests 0x6 )
make_named_test(hipMemcpy "hipMemcpy-multithreaded" --tests 0x8 )

# Debug synchronization, then enable.
#make_named_test(hipMemcpy_simple "hipMemcpyAsync-simple" --async)

make_test(hipHostAlloc " ")
#make_test(hipMemcpyAsync  " " )
# BS- comment out since test appears broken - asks for device pointer but pointer was never allocated.
#make_test(hipHostGetFlags " ")
make_test(hipHcc  " " )
make_test(hipHostRegister " ")
make_test(hipStreamL5 " ")
make_test(hipRandomMemcpyAsync " ")
#make_test(hipAPIStreamEnable " ")
#make_test(hipAPIStreamDisable " ")
make_test(hipFuncSetDeviceFlags " ")
make_test(hipFuncGetDevice " ")
make_test(hipFuncDeviceSynchronize " ")
make_test(hipTestMemcpyPin " ")
make_named_test (hipMultiThreadDevice "hipMultiThreadDevice-serial" --tests 0x1) 
make_named_test (hipMultiThreadDevice "hipMultiThreadDevice-pyramid" --tests 0x4) 
make_named_test (hipMultiThreadDevice "hipMultiThreadDevice-nearzero" --tests 0x10) 

if (${HIP_MULTI_GPU})
    make_test(hipPeerToPeer_simple " ")                  # use current device for copy, this fails. 
    make_test(hipPeerToPeer_simple --memcpyWithPeer)  
    make_test(hipPeerToPeer_simple --mirrorPeers)    # mirror mapping: test to ensure mirror doesn't destroy orig mapping.

endif()

if (${HIP_PLATFORM} STREQUAL "hcc")
    make_test(hip_test_ldg " " )
    make_test(hipMemoryAllocate " ")
    make_test(hipFuncSetDevice " ")
endif()

make_hipify_test(specialFunc.cu )

make_test(hipDynamicShared " ")
