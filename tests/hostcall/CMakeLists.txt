include(${HIP_SRC_PATH}/cmake/FindHIP.cmake)
message(STATUS "Using hipcc for hostcall tests: ${HIP_HIPCC_EXECUTABLE}")

# FIXME: The NAMES argument should not be necessary, but the default
# HSA_PATH /opt/rocm/hsa/lib only contains a versioned
# library. HSA_PATH should either point to /opt/rocm where lib/
# contains an unversioned symlink, or /opt/rocm/hsa packaging should
# be updated.
#
# This also requires extra work to detect the filename of the library
# and pass it as "-l:filename" to the linker.
find_library(
    HSA_LIBRARY hsa-runtime64
    NAMES libhsa-runtime64.so libhsa-runtime64.so.1
    PATHS ${HSA_PATH}
    PATH_SUFFIXES lib NO_DEFAULT_PATH)
if (EXISTS ${HSA_LIBRARY})
    get_filename_component(HSA_LIB_DIR ${HSA_LIBRARY} DIRECTORY)
    get_filename_component(HSA_LIBRARY ${HSA_LIBRARY} NAME)
    message(STATUS "Using HSA runtime for hostcall tests: ${HSA_LIB_DIR}")
else()
    message(FATAL_ERROR "*** Cannot find HSA libraries in ${HSA_PATH}")
endif()

find_path(HSA_HEADER hsa/hsa.h PATHS ${HSA_PATH} PATH_SUFFIXES include NO_DEFAULT_PATH)
find_path(HSA_HEADER hsa/hsa.h PATHS /opt/rocm PATH_SUFFIXES include)
if (NOT EXISTS ${HSA_HEADER})
    message(FATAL_ERROR "*** Cannot find HSA headers in ${HSA_PATH}")
else()
    message(STATUS "Using HSA headers for hostcall tests: ${HSA_HEADER}")
endif()

set(HOSTCALL_TEST_INC ${CMAKE_CURRENT_LIST_DIR})

function(hostcall_build_test _testname filename)
    get_filename_component(prefix ${CMAKE_CURRENT_LIST_DIR} NAME)
    get_filename_component(basename_we ${filename} NAME_WE)
    get_filename_component(basename ${filename} NAME)
    set(testname hostcall-${prefix}-${basename_we})
    set(${_testname} ${testname} PARENT_SCOPE)

    add_custom_target(
        ${testname}
        COMMAND ${HIP_HIPCC_EXECUTABLE} -g
        -I${HSA_HEADER} -I${HOSTCALL_TEST_INC}
        -I$<TARGET_PROPERTY:amd_hostcall,INTERFACE_INCLUDE_DIRECTORIES>
        -L$<TARGET_FILE_DIR:amd_hostcall> -L${HSA_LIB_DIR}
        -lpthread -l:$<TARGET_FILE_NAME:amd_hostcall> -l:${HSA_LIBRARY}
        ${filename} -o ${testname}
        COMMENT "Building ${testname}")

    add_dependencies(hostcall_tests ${testname})
endfunction()

function(hostcall_add_test _testname filename)
    hostcall_build_test(testname ${filename})
    add_test(${testname} ${testname})
    set(${_testname} ${testname} PARENT_SCOPE)
endfunction()

function(hostcall_add_tests files)
    foreach(file ${files})
        hostcall_add_test(testname ${file})
    endforeach()
endfunction()

add_subdirectory(basic EXCLUDE_FROM_ALL)
add_subdirectory(device EXCLUDE_FROM_ALL)
