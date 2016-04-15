
# findHCC does not currently address versioning, i.e.
# a rich directory structure where version number is a subdirectory under root
# Also, supported only on UNIX 64 bit systems.

if(UNIX)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)  
  
		find_library(HSA_LIBRARY
			NAMES  hsa-runtime64
			PATHS
			  ENV HSA_PATH
			  /opt/rocm/hsa
			PATH_SUFFIXES
			  lib)
		
		if( NOT DEFINED ENV{HSA_PATH} )
			set( ENV{HSA_PATH} /opt/rocm/hsa)
		endif()
		
		find_program(HCC
			NAMES  hcc
			PATHS
				ENV HCC_PATH
				/opt/rocm/hcc
			PATH_SUFFIXES
				/bin)

		if( NOT DEFINED  ENV{HCC_PATH} )
			set( ENV{HCC_PATH} /opt/rocm/hcc)
		endif()
		
# this is now dynamic
#		find_library(AMP_LIBRARY
#			NAMES  mcwamp
#			PATHS
#				ENV NCC_PATH
#				/opt/rocm/hcc
#			PATH_SUFFIXES
#				/lib)
				
		find_path(HCC_INCLUDE_DIR
			NAMES
				hc.hpp
			PATHS
				ENV NCC_PATH
				/opt/rocm/hcc
			PATH_SUFFIXES
				/include)				
				
			  

		set(HSA_LIBRARIES ${HSA_LIBRARY})
		#set(HCC_LIBRARIES ${AMP_LIBRARY})
		set(HCC_INCLUDE_DIRS ${HCC_INCLUDE_DIR})

		include(FindPackageHandleStandardArgs)
		find_package_handle_standard_args(
		  HCC
		  FOUND_VAR HCC_FOUND
		  REQUIRED_VARS HSA_LIBRARIES HCC_INCLUDE_DIRS HCC)

		mark_as_advanced(
		  HSA_LIBRARIES
		  HCC_INCLUDE_DIRS
		)

	else()
		message(SEND_ERROR "HCC is currently supported only on 64 bit UNIX platforms")
	endif()
else()
	message(SEND_ERROR "HCC is currently supported on unix platforms")
endif()
