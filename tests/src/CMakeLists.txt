cmake_minimum_required (VERSION 2.6)
project (HIP_Unit_Tests)
include(CTest)

include_directories( ${PROJECT_SOURCE_DIR}/include )

# The version number.
set (HIP_Unit_Test_VERSION_MAJOR 1)
set (HIP_Unit_Test_VERSION_MINOR 0)

set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
	set (HIP_PATH ../..)
endif()

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)

MESSAGE ("HIP_PATH=" ${HIP_PATH})

if (${HIP_PLATFORM} STREQUAL "hcc")
	MESSAGE ("HIP_PLATFORM=hcc")
    set (HSA_PATH $ENV{HSA_PATH})
    if (NOT DEFINED HSA_PATH)
        set (HSA_PATH /opt/hsa)
    endif()

	#---
	# Add HSA library:
	add_library(hsa-runtime64 SHARED IMPORTED)
	set_property(TARGET hsa-runtime64 PROPERTY IMPORTED_LOCATION "${HSA_PATH}/lib/libhsa-runtime64.so") 

	#These includes are used for all files.
	#Include HIP and HC since the tests need both of these:
	#Note below HSA path is surgically included only where necessary.
	include_directories(${HIP_PATH}/include)

	# hip_hcc.o:
	add_library(hip_hcc OBJECT ${HIP_PATH}/src/hip_hcc.cpp)
	target_include_directories(hip_hcc PRIVATE ${HSA_PATH}/include)


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
        add_executable (${exe} ${cpp} ${ARGN} $<TARGET_OBJECTS:test_common> $<TARGET_OBJECTS:hip_hcc> )
    else()
        add_executable (${exe} ${cpp} ${ARGN} $<TARGET_OBJECTS:test_common> )
    endif() 
endmacro()


macro (make_test exe )
  string (REPLACE " " "" smush_args ${ARGN})
  set (testname ${exe}${smush_args}.tst)
  add_test (NAME ${testname} 
      COMMAND ${PROJECT_BINARY_DIR}/${exe} ${ARGN}
            )
  set_tests_properties (${testname}
      PROPERTIES PASS_REGULAR_EXPRESSION "PASSED"
    )
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

make_hip_executable (hip_ballot hip_ballot.cpp)
make_hip_executable (hip_anyall hip_anyall.cpp) 
make_hip_executable (hip_clz hip_clz.cpp) 
make_hip_executable (hip_brev hip_brev.cpp) 
make_hip_executable (hip_ffs hip_ffs.cpp) 
make_hip_executable (hipGetDeviceAttribute hipGetDeviceAttribute.cpp) 
make_hip_executable (hipEnvVar hipEnvVar.cpp) 
make_hip_executable (hipEnvVarDriver hipEnvVarDriver.cpp) 
make_hip_executable (hipMemcpy hipMemcpy.cpp) 
make_hip_executable (hipMemcpyAsync hipMemcpyAsync.cpp) 
make_hip_executable (hipMemset hipMemset.cpp) 
make_hip_executable (hipEventRecord hipEventRecord.cpp) 
make_hip_executable (hipLanguageExtensions hipLanguageExtensions.cpp) 
make_hip_executable (hipGridLaunch hipGridLaunch.cpp) 
make_hip_executable (hipSimpleAtomicsTest hipSimpleAtomicsTest.cpp)
make_hip_executable (hipMathFunctionsHost hipMathFunctions.cpp hipSinglePrecisionMathHost.cpp hipDoublePrecisionMathHost.cpp)
make_hip_executable (hipMathFunctionsDevice hipMathFunctions.cpp hipSinglePrecisionMathDevice.cpp hipDoublePrecisionMathDevice.cpp)
make_hip_executable (hipIntrinsics hipMathFunctions.cpp hipSinglePrecisionIntrinsics.cpp hipDoublePrecisionIntrinsics.cpp hipIntegerIntrinsics.cpp)
make_hip_executable (hipPointerAttrib hipPointerAttrib.cpp) 
make_hip_executable (hipMultiThreadStreams1 hipMultiThreadStreams1.cpp)
make_hip_executable (hipMultiThreadStreams2 hipMultiThreadStreams2.cpp) 

make_hip_executable (hipStreamL5 hipStreamL5.cpp)
target_link_libraries(hipMathFunctionsHost m)

make_test(hip_anyall " " )
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
make_test(hipPointerAttrib " " )
make_test(hipMultiThreadStreams1 " " )
make_test(hipMultiThreadStreams2 " " )

make_test(hipMemcpy  " " )
make_test(hipMemcpyAsync  " " )

# Isolate some tests which only run on HCC platform, for now:
if (${HIP_PLATFORM} STREQUAL "hcc")
make_hip_executable (hip_popc hip_popc.cpp) 
make_hip_executable (hipHcc hipHcc.cpp) 
make_test(hip_popc " " )
make_test(hipHcc  " " )
# builds but fails at runtime:
make_test(hip_ballot " " )
endif()

make_test(hipStreamL5 " ")

make_hipify_test(specialFunc.cu )
