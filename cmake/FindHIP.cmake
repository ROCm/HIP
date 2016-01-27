
find_package(CUDA)

if( CUDA_INCLUDE_DIRS AND CUDA_VERSION AND CUDA_NVCC_EXECUTABLE)
	message(STATUS "CUDA_VERSION = ${CUDA_VERSION}")
	message(STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
	message(STATUS "CUDA_NVCC_EXECUTABLE = ${CUDA_NVCC_EXECUTABLE}")
	
    set( HIP_PLATFORM "nvcc" )
	
	#export the environment variable, so that HIPCC can find it.
    set(ENV{HIP_PLATFORM} nvcc)
	
else()
	find_package(HCC)
	
	message(STATUS "HCC_FOUND = ${HCC_FOUND}")
	message(STATUS "HCC = ${HCC}")
	message(STATUS "HCC_INCLUDE_DIRS = ${HCC_INCLUDE_DIRS}")
	message(STATUS "HSA_LIBRARIES = ${HSA_LIBRARIES}")

	if( ${HCC_FOUND} STREQUAL  "TRUE" )
	
	# This directory is hip/cmake! HIP_PATH should be one directory up!
		set (HIP_PATH $ENV{HIP_PATH})
		if (NOT DEFINED HIP_PATH)
			set (HIP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
			set( ENV{HIP_PATH} ${HIP_PATH})
		endif()

		message(STATUS "ENV HIP_PATH = $ENV{HIP_PATH}")
		
		find_program(HIPCC
			NAMES  hipcc
			PATHS
				ENV HIP_PATH
			PATH_SUFFIXES
				/bin)
			
			message(STATUS "HIPCC = ${HIPCC}")
			
		if( DEFINED HIPCC)
		
            set( HIP_PLATFORM "hcc" )		
			#export the environment variable, so that HIPCC can find it.
            set(ENV{HIP_PLATFORM} "hcc")	
			set (CMAKE_CXX_COMPILER ${HIPCC})
			
		else()
			message(SEND_ERROR "Did not find HIPCC")
		endif()
	else()
		message(SEND_ERROR "hcc not found")
	endif()
	
endif()
