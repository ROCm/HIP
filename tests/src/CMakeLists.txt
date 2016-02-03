cmake_minimum_required (VERSION 2.6)
project (HIP_Unit_Tests)
include(CTest)

include_directories( ${PROJECT_SOURCE_DIR}/include )

# The version number.
set (HIP_Unit_Test_VERSION_MAJOR 1)
set (HIP_Unit_Test_VERSION_MINOR 0)

set (CUDA_PATH $ENV{CUDA_PATH})
if (NOT DEFINED CUDA_PATH)
	set( CUDA_PATH /usr/local/cuda)
endif()

set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
	set (HIP_PATH ../..)
endif()

set (HIP_PLATFORM $ENV{HIP_PLATFORM})
if (NOT DEFINED HIP_PLATFORM) 
	if (EXISTS $CUDA_PATH)	
        set (HIP_PLATFORM nvcc)
	else()
        set (HIP_PLATFORM hcc)
	endif()
endif()



if (${HIP_PLATFORM} STREQUAL "hcc")
	MESSAGE ("HCC")
	set (HC_PATH ${HIP_PATH}/hc)
	set (HSA_PATH /opt/hsa)


	#---
	# Add HSA library:
	add_library(hsa-runtime64 SHARED IMPORTED)
	set_property(TARGET hsa-runtime64 PROPERTY IMPORTED_LOCATION "${HSA_PATH}/lib/libhsa-runtime64.so") 

	#These includes are used for all files.
	#Include HIP and HC since the tests need both of these:
	#Note below HSA path is surgically included only where necessary.
	include_directories(${HIP_PATH}/include ${HC_PATH}/include)

	# hip_hcc.o:
	add_library(hip_hcc OBJECT ${HIP_PATH}/src/hip_hcc.cpp)
	target_include_directories(hip_hcc PRIVATE ${HSA_PATH}/include)


elseif (${HIP_PLATFORM} STREQUAL "nvcc")
	MESSAGE ("NVCC")
	# NVCC does not not support -rdynamic option
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS )
	set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS )
else()
    MESSAGE ("UNKNOWN HIP_PLATFORM=" ${HIP_PLATFORM})
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


#set(CMAKE_INSTALL_PREFIX "./install")
#install (TARGETS hipMemset DESTINATION bin)
#install (TARGETS hipEventRecord DESTINATION bin)

make_hip_executable (hip_ballot hip_ballot.cpp)
make_hip_executable (hip_anyall hip_anyall.cpp) 
make_hip_executable (hip_popc hip_popc.cpp) 
make_hip_executable (hip_clz hip_clz.cpp) 
make_hip_executable (hip_brev hip_brev.cpp) 
make_hip_executable (hip_ffs hip_ffs.cpp) 
make_hip_executable (hipMemset hipMemset.cpp) 
make_hip_executable (hipMemcpy hipMemcpy.cpp) 
make_hip_executable (hipEventRecord hipEventRecord.cpp) 
make_hip_executable (hipLanguageExtensions hipLanguageExtensions.cpp) 
make_hip_executable (hipGridLaunch hipGridLaunch.cpp) 
make_hip_executable (hipHcc hipHcc.cpp) 
make_hip_executable (hipSimpleAtomicsTest hipSimpleAtomicsTest.cpp)
make_hip_executable (hipMathFunctionsHost hipMathFunctions.cpp hipSinglePrecisionMathHost.cpp hipDoublePrecisionMathHost.cpp)
make_hip_executable (hipMathFunctionsDevice hipMathFunctions.cpp hipSinglePrecisionMathDevice.cpp hipDoublePrecisionMathDevice.cpp)
make_hip_executable (hipIntrinsics hipMathFunctions.cpp hipSinglePrecisionIntrinsics.cpp hipDoublePrecisionIntrinsics.cpp hipIntegerIntrinsics.cpp)
target_link_libraries(hipMathFunctionsHost m)

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
make_test(hipMemset --N 500M  --memsetval 0xa6 )  # big copy
make_test(hipGridLaunch " " )

make_test(hipMemcpy  " " )

make_test(hipHcc  " " )

make_hipify_test(specialFunc.cu )
