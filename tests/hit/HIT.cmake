# Copyright (c) 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
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

include(CTest)
find_package(HIP REQUIRED)

set(HIP_CTEST_CONFIG_DEFAULT "default")
set(HIP_CTEST_CONFIG_PERFORMANCE "performance")
set(HIP_LIB_TYPE "shared")
if (NOT ${BUILD_SHARED_LIBS})
    set(HIP_LIB_TYPE "static")
endif()
message(STATUS "HIP runtime lib type - ${HIP_LIB_TYPE}")
message(STATUS "CMAKE_TESTING_TOOL: ${CMAKE_TESTING_TOOL}")
#-------------------------------------------------------------------------------
# Helper macro to parse BUILD instructions
macro(PARSE_BUILD_COMMAND _target _sources _hipcc_options _clang_options _nvcc_options _link_options _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type _depends _dir)
    set(${_target})
    set(${_sources})
    set(${_hipcc_options})
    set(${_clang_options})
    set(${_nvcc_options})
    set(${_link_options})
    set(${_exclude_platforms})
    set(${_exclude_runtime})
    set(${_exclude_compiler})
    set(${_exclude_lib_type})
    set(${_depends})

    set(_target_found FALSE)
    set(_flag "")

    foreach(arg ${ARGN})
        if(NOT _target_found)
            set(_target_found TRUE)
            set(${_target} ${arg})
        elseif("x${arg}" STREQUAL "xHIPCC_OPTIONS"
            OR "x${arg}" STREQUAL "xCLANG_OPTIONS"
            OR "x${arg}" STREQUAL "xNVCC_OPTIONS"
            OR "x${arg}" STREQUAL "xLINK_OPTIONS"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_PLATFORM"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_RUNTIME"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_COMPILER"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE"
            OR "x${arg}" STREQUAL "xDEPENDS")
            set(_flag ${arg})
        elseif("x${_flag}" STREQUAL "xHIPCC_OPTIONS")
            list(APPEND ${_hipcc_options} ${arg})
        elseif("x${_flag}" STREQUAL "xCLANG_OPTIONS")
            list(APPEND ${_clang_options} ${arg})
        elseif("x${_flag}" STREQUAL "xNVCC_OPTIONS")
            list(APPEND ${_nvcc_options} ${arg})
        elseif("x${_flag}" STREQUAL "xLINK_OPTIONS")
            list(APPEND ${_link_options} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_PLATFORM")
            list(APPEND ${_exclude_platforms} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_RUNTIME")
            list(APPEND ${_exclude_runtime} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_COMPILER")
            list(APPEND ${_exclude_compiler} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE")
            list(APPEND ${_exclude_lib_type} ${arg})
        elseif("x${_flag}" STREQUAL "xDEPENDS")
            list(APPEND ${_depends} ${arg})
        else()
            list(APPEND ${_sources} "${_dir}/${arg}")
        endif()
    endforeach()
endmacro()

# Helper macro to parse CUSTOM BUILD instructions
macro(PARSE_CUSTOMBUILD_COMMAND _target _buildcmd _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type _depends)
    set(${_target})
    set(${_buildcmd})
    set(${_exclude_platforms})
    set(${_exclude_runtime})
    set(${_exclude_compiler})
    set(${_exclude_lib_type})
    set(${_depends})

    set(_target_found FALSE)
    set(_flag "")

    foreach(arg ${ARGN})
        if(NOT _target_found)
            set(_target_found TRUE)
            set(${_target} ${arg})
        elseif("x${arg}" STREQUAL "xEXCLUDE_HIP_PLATFORM"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_RUNTIME"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_COMPILER"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE"
            OR "x${arg}" STREQUAL "xDEPENDS")
            set(_flag ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_PLATFORM")
            list(APPEND ${_exclude_platforms} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_RUNTIME")
            list(APPEND ${_exclude_runtime} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_COMPILER")
            list(APPEND ${_exclude_compiler} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE")
            list(APPEND ${_exclude_lib_type} ${arg})
        elseif("x${_flag}" STREQUAL "xDEPENDS")
            list(APPEND ${_depends} ${arg})
        else()
            list(APPEND ${_buildcmd} ${arg})  # always before exclude lists
        endif()
    endforeach()
endmacro()

# Helper macro to parse command part of CUSTOM BUILD instructions
macro(PARSE_CUSTOMBUILD_COMMAND_PART _compiler _target _target_type _sources _options)
    set(${_compiler})
    set(${_target})
    set(${_target_type} "EXECUTABLE")
    set(${_sources})
    set(${_options})
    set(_compiler_found FALSE)
    set(_target_found FALSE)

    foreach(arg ${ARGN})
        if(NOT _compiler_found)
            set(_compiler_found TRUE)
            set(${_compiler} ${arg})
        elseif("x${arg}" STREQUAL "x-o")
            set(_target_found TRUE)
        elseif(_target_found)
            set(${_target} ${arg})
            set(_target_found FALSE)
        elseif("x${arg}" STREQUAL "x-c" OR "x${arg}" STREQUAL "x--genco")
            set(${_target_type} "OBJECT")
            list(APPEND ${_options} ${arg})
        elseif("x${arg}" STREQUAL "x-shared")
            # Note: Currently all directed_tests are linux based.
            set(${_target_type} "SHARED")
            list(APPEND ${_options} ${arg})
        elseif("x${arg}" MATCHES "^x-I")
            # -I
            list(APPEND ${_options} ${arg})
        elseif("x${arg}" MATCHES "^x.*\.cpp$")
            # cpp file
            list(APPEND ${_sources} ${arg})
        elseif("x${arg}" MATCHES "^x.*\.c$")
            # c file
            list(APPEND ${_sources} ${arg})
        else()
            list(APPEND ${_options} ${arg})
        endif()
    endforeach()
endmacro()

# Helper macro to parse TEST instructions
macro(PARSE_TEST_COMMAND _target _arguments _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type)
    set(${_target})
    set(${_arguments} " ")
    set(${_exclude_platforms})
    set(${_exclude_runtime})
    set(${_exclude_compiler})
    set(${_exclude_lib_type})

    set(_target_found FALSE)
    set(_flag "")

    foreach(arg ${ARGN})
        if(NOT _target_found)
            set(_target_found TRUE)
            set(${_target} ${arg})
        elseif("x${arg}" STREQUAL "xEXCLUDE_HIP_PLATFORM"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_RUNTIME"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_COMPILER"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE")
            set(_flag ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_PLATFORM")
            list(APPEND ${_exclude_platforms} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_RUNTIME")
            list(APPEND ${_exclude_runtime} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_COMPILER")
            list(APPEND ${_exclude_compiler} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE")
            list(APPEND ${_exclude_lib_type} ${arg})
        else()
            list(APPEND ${_arguments} ${arg}) # always before exclude lists
        endif()
    endforeach()
endmacro()

# Helper macro to parse TEST_NAMED instructions
macro(PARSE_TEST_NAMED_COMMAND _target _testname _arguments _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type)
    set(${_target})
    set(${_arguments} " ")
    set(${_exclude_platforms})
    set(${_exclude_runtime})
    set(${_exclude_compiler})
    set(${_exclude_lib_type})

    set(_target_found FALSE)
    set(_testname_found FALSE)
    set(_flag "")

    foreach(arg ${ARGN})
        if(NOT _target_found)
            set(_target_found TRUE)
            set(${_target} ${arg})
        elseif(NOT _testname_found)
            set(_testname_found TRUE)
            set(${_testname} ${arg})
        elseif("x${arg}" STREQUAL "xEXCLUDE_HIP_PLATFORM"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_RUNTIME"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_COMPILER"
            OR "x${arg}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE")
            set(_flag ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_PLATFORM")
            list(APPEND ${_exclude_platforms} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_RUNTIME")
            list(APPEND ${_exclude_runtime} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_COMPILER")
            list(APPEND ${_exclude_compiler} ${arg})
        elseif("x${_flag}" STREQUAL "xEXCLUDE_HIP_LIB_TYPE")
            list(APPEND ${_exclude_lib_type} ${arg})
        else()
            list(APPEND ${_arguments} ${arg}) # always before exclude lists
        endif()
    endforeach()
endmacro()

# Helper macro to insert key/value pair into "hashmap"
macro(INSERT_INTO_MAP _map _key _value)
    set("${_map}_${_key}" "${_value}")
endmacro()

# Helper macro to read key/value pair from "hashmap"
macro(READ_FROM_MAP _map _key _value)
    set(${_value} "${${_map}_${_key}}")
endmacro()


# Helper macro to generate a test
macro(GENERATE_TEST _config testname cmdline)
    set(TEST_CMD_LINE ${cmdline} ${ARGN})
    if(${_config} STREQUAL ${HIP_CTEST_CONFIG_DEFAULT})
        add_test(NAME ${testname} COMMAND ${TEST_CMD_LINE})
    else()
        add_test(NAME ${testname} CONFIGURATIONS ${_config} COMMAND ${TEST_CMD_LINE})
    endif()
    set_tests_properties(${testname} PROPERTIES PASS_REGULAR_EXPRESSION "PASSED" ENVIRONMENT HIP_PATH=${HIP_ROOT_DIR})
    set_tests_properties(${testname} PROPERTIES SKIP_RETURN_CODE 127 ENVIRONMENT HIP_PATH=${HIP_ROOT_DIR})
endmacro()

# Helper macro to create a test
macro(MAKE_NAMED_TEST _config exe testname)
    # to generate hip original test
    set(TEST_CMD_LINE ${PROJECT_BINARY_DIR}/${exe} ${ARGN})
    generate_test(${_config} ${testname} ${TEST_CMD_LINE})

    # to generate test with tool enabled
    if(DEFINED CMAKE_TESTING_TOOL)
        # arguments passing to the testing tool
        # <source dir>, <build dir>, <test name>, <test args...>
        set(TOOL_CMD_LINE ${CMAKE_TESTING_TOOL} ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR} ${TEST_CMD_LINE})
        generate_test(${_config} ${testname}.prof ${TOOL_CMD_LINE})
    endif()
endmacro()

# Helper macro to create a test with default name
macro(MAKE_TEST _config exe)
    string(REPLACE " " "" smush_args ${ARGN})
    set(testname ${exe}${smush_args}.tst)
    make_named_test(${_config} ${exe} ${testname} ${ARGN})
endmacro()
#-------------------------------------------------------------------------------

# Macro: HIT_ADD_FILES used to scan+add multiple files for testing.
file(GLOB HIP_LIB_FILES ${HIP_PATH}/lib/*)
macro(HIT_ADD_FILES _config _dir _label _parent)
    foreach (file ${ARGN})
        # Build tests
        execute_process(COMMAND ${HIP_SRC_PATH}/tests/hit/parser --buildCMDs ${file}
            OUTPUT_VARIABLE _contents
            ERROR_QUIET
            WORKING_DIRECTORY ${_dir}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "\n" ";" _contents "${_contents}")
        foreach(_cmd ${_contents})
            string(REGEX REPLACE " " ";" _cmd "${_cmd}")
            parse_build_command(_target _sources _hipcc_options _clang_options _nvcc_options _link_options _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type _depends ${_dir} ${_cmd})
            string(REGEX REPLACE "/" "." target ${_label}/${_target})
            if("all" IN_LIST _exclude_platforms OR ${HIP_PLATFORM} IN_LIST _exclude_platforms)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(NOT _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(NOT _exclude_compiler AND ${HIP_RUNTIME} IN_LIST _exclude_runtime)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(${HIP_RUNTIME} IN_LIST _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(${HIP_LIB_TYPE} IN_LIST _exclude_lib_type)
                insert_into_map("_exclude" "${target}" TRUE)
            else()
                set_source_files_properties(${_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
                hip_reset_flags()
                hip_add_executable(${target} ${_sources} HIPCC_OPTIONS ${_hipcc_options} CLANG_OPTIONS ${_clang_options} NVCC_OPTIONS ${_nvcc_options} EXCLUDE_FROM_ALL)
                target_link_libraries(${target} PRIVATE ${_link_options})
                set_target_properties(${target} PROPERTIES OUTPUT_NAME ${_target} RUNTIME_OUTPUT_DIRECTORY ${_label} LINK_DEPENDS "${HIP_LIB_FILES}")
                add_dependencies(${_parent} ${target})
                foreach(_dependency ${_depends})
                    string(REGEX REPLACE "/" "." _dependency ${_label}/${_dependency})
                    add_dependencies(${target} ${_dependency})
                endforeach()
            endif()
        endforeach()

        # Custom build commands
        execute_process(COMMAND ${HIP_SRC_PATH}/tests/hit/parser --customBuildCMDs ${file}
            OUTPUT_VARIABLE _contents
            ERROR_QUIET
            WORKING_DIRECTORY ${_dir}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "\n" ";" _contents "${_contents}")
        string(REGEX REPLACE "%hc" "${HIP_HIPCC_EXECUTABLE}" _contents "${_contents}")
        string(REGEX REPLACE "%hip-path" "${HIP_ROOT_DIR}" _contents "${_contents}")
        string(REGEX REPLACE "%rocm-path" "${ROCM_PATH}" _contents "${_contents}")
        string(REGEX REPLACE "%cc" "/usr/bin/cc" _contents "${_contents}")
        string(REGEX REPLACE "%cxx" "/usr/bin/c++" _contents "${_contents}")
        string(REGEX REPLACE "%S" ${_dir} _contents "${_contents}")
        string(REGEX REPLACE "%T" ${_label} _contents "${_contents}")
        foreach(_cmd ${_contents})
            string(REGEX REPLACE " " ";" _cmd "${_cmd}")
            parse_custombuild_command(_target _buildcmd _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type _depends ${_cmd})
            string(REGEX REPLACE "/" "." target ${_label}/${_target})
            if("all" IN_LIST _exclude_platforms OR ${HIP_PLATFORM} IN_LIST _exclude_platforms)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(NOT _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(NOT _exclude_compiler AND ${HIP_RUNTIME} IN_LIST _exclude_runtime)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(${HIP_RUNTIME} IN_LIST _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
                insert_into_map("_exclude" "${target}" TRUE)
            elseif(${HIP_LIB_TYPE} IN_LIST _exclude_lib_type)
                insert_into_map("_exclude" "${target}" TRUE)
            else()
                parse_custombuild_command_part(_compiler _target_r _target_type _sources _options ${_buildcmd})
                string(REGEX REPLACE ";" " " _buildcmd "${_buildcmd}")
                string(REGEX REPLACE ";" " " _options "${_options}")

                set(CHOICE_FLAG "${HIP_LIB_TYPE}" STREQUAL "static" AND "${_compiler}" MATCHES "hipcc$"
                    AND "${HIP_RUNTIME}" STREQUAL "rocclr" AND "${HIP_COMPILER}" STREQUAL "clang")
                if (${CHOICE_FLAG} AND "${_target_type}" STREQUAL "EXECUTABLE")
                        # message(STATUS "hip_add_executable*:_target_r= ${_target_r} --- target= ${target} --- _sources=${_sources} --- _options=${_options}")
                        set_source_files_properties(${_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
                        hip_reset_flags()
                        hip_add_executable(${target} ${_sources} HIPCC_OPTIONS ${_options} EXCLUDE_FROM_ALL)
                        set_target_properties(${target} PROPERTIES OUTPUT_NAME ${_target_r} RUNTIME_OUTPUT_DIRECTORY "." LINK_DEPENDS "${HIP_LIB_FILES}")
                elseif(${CHOICE_FLAG} AND "${_target_type}" STREQUAL "SHARED")
                        # message(STATUS "hip_add_library*:_target_r= ${_target_r} --- target= ${target} --- _sources=${_sources} --- _options=${_options}")
                        set_source_files_properties(${_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
                        hip_reset_flags()
                        hip_add_library(${target} ${_sources} HIPCC_OPTIONS ${_options} EXCLUDE_FROM_ALL ${_target_type})
                        set_target_properties(${target} PROPERTIES OUTPUT_NAME ${_target_r} RUNTIME_OUTPUT_DIRECTORY "." LINK_DEPENDS "${HIP_LIB_FILES}" PREFIX "" SUFFIX "")
                else()
                    # message(STATUS "add_custom_target*: target= ${target}  _buildcmd= ${_buildcmd}")
                    add_custom_target(${target} COMMAND sh -c "${_buildcmd}")
                endif()
                add_dependencies(${_parent} ${target})
                foreach(_dependency ${_depends})
                    string(REGEX REPLACE "/" "." _dependency ${_label}/${_dependency})
                    add_dependencies(${target} ${_dependency})
                endforeach()
            endif()
        endforeach()

        # Add tests
        execute_process(COMMAND ${HIP_SRC_PATH}/tests/hit/parser --testCMDs ${file}
            OUTPUT_VARIABLE _contents
            ERROR_QUIET
            WORKING_DIRECTORY ${_dir}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "\n" ";" _contents "${_contents}")
        foreach(_cmd ${_contents})
            string(REGEX REPLACE " " ";" _cmd "${_cmd}")
            parse_test_command(_target _arguments _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type ${_cmd})
            string(REGEX REPLACE "/" "." target ${_label}/${_target})
            read_from_map("_exclude" "${target}" _exclude_test_from_build)
            if("all" IN_LIST _exclude_platforms OR ${HIP_PLATFORM} IN_LIST _exclude_platforms)
            elseif(NOT _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
            elseif(NOT _exclude_compiler AND ${HIP_RUNTIME} IN_LIST _exclude_runtime)
            elseif(${HIP_RUNTIME} IN_LIST _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
            elseif(${HIP_LIB_TYPE} IN_LIST _exclude_lib_type)
            elseif(_exclude_test_from_build STREQUAL TRUE)
            else()
                make_test(${_config} ${_label}/${_target} ${_arguments})
            endif()
        endforeach()

        # Add named tests
        execute_process(COMMAND ${HIP_SRC_PATH}/tests/hit/parser --testNamedCMDs ${file}
            OUTPUT_VARIABLE _contents
            ERROR_QUIET
            WORKING_DIRECTORY ${_dir}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "\n" ";" _contents "${_contents}")
        foreach(_cmd ${_contents})
            string(REGEX REPLACE " " ";" _cmd "${_cmd}")
            parse_test_named_command(_target _testname _arguments _exclude_platforms _exclude_runtime _exclude_compiler _exclude_lib_type ${_cmd})
            string(REGEX REPLACE "/" "." target ${_label}/${_target})
            read_from_map("_exclude" "${target}" _exclude_test_from_build)
            if("all" IN_LIST _exclude_platforms OR ${HIP_PLATFORM} IN_LIST _exclude_platforms)
            elseif(NOT _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
            elseif(NOT _exclude_compiler AND ${HIP_RUNTIME} IN_LIST _exclude_runtime)
            elseif(${HIP_RUNTIME} IN_LIST _exclude_runtime AND ${HIP_COMPILER} IN_LIST _exclude_compiler)
            elseif(${HIP_LIB_TYPE} IN_LIST _exclude_lib_type)
            elseif(_exclude_test_from_build STREQUAL TRUE)
            else()
                make_named_test(${_config} ${_label}/${_target} ${_label}/${_testname}.tst ${_arguments})
            endif()
        endforeach()
    endforeach()
endmacro()

# Macro: HIT_ADD_DIRECTORY to scan+add all files in a directory for testing
macro(HIT_ADD_DIRECTORY _dir _label)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${_label} WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    string(REGEX REPLACE "/" "." _parent ${_label})
    add_custom_target(${_parent})
    file(GLOB files "${_dir}/*.c*")
    hit_add_files(${HIP_CTEST_CONFIG_DEFAULT} ${_dir} ${_label} ${parent} ${files})
endmacro()

# Macro: HIT_ADD_DIRECTORY_RECURSIVE to scan+add all files in a directory+subdirectories for testing
macro(HIT_ADD_DIRECTORY_RECURSIVE _config _dir _label)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${_label} WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    string(REGEX REPLACE "/" "." _parent ${_label})
    add_custom_target(${_parent})
    if(${ARGC} EQUAL 4)
        add_dependencies(${ARGV3} ${_parent})
    endif()
    file(GLOB children RELATIVE ${_dir} ${_dir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${_dir}/${child})
            list(APPEND dirlist ${child})
        else()
            hit_add_files(${_config} ${_dir} ${_label} ${_parent} ${child})
        endif()
    endforeach()
    foreach(child ${dirlist})
        string(REGEX REPLACE "/" "." _parent ${_label})
        hit_add_directory_recursive(${_config} ${_dir}/${child} ${_label}/${child} ${_parent})
    endforeach()
endmacro()

# vim: ts=4:sw=4:expandtab:smartindent
