include(CTest)
find_package(HIP REQUIRED)

#-------------------------------------------------------------------------------
# Helper macro to parse BUILD instructions
macro(PARSE_BUILD_COMMAND _target _sources _hipcc_options _hcc_options _nvcc_options _exclude_platforms _dir)
    set(${_target})
    set(${_sources})
    set(${_hipcc_options})
    set(${_hcc_options})
    set(${_nvcc_options})
    set(${_exclude_platforms})
    set(_target_found FALSE)
    set(_hipcc_options_found FALSE)
    set(_hcc_options_found FALSE)
    set(_nvcc_options_found FALSE)
    set(_exclude_platforms_found FALSE)
    foreach(arg ${ARGN})
        if(NOT _target_found)
            set(_target_found TRUE)
            set(${_target} ${arg})
        elseif("x${arg}" STREQUAL "xHIPCC_OPTIONS")
            set(_hipcc_options_found TRUE)
            set(_hcc_options_found FALSE)
            set(_nvcc_options_found FALSE)
            set(_exclude_platforms_found FALSE)
        elseif("x${arg}" STREQUAL "xHCC_OPTIONS")
            set(_hipcc_options_found FALSE)
            set(_hcc_options_found TRUE)
            set(_nvcc_options_found FALSE)
            set(_exclude_platforms_found FALSE)
        elseif("x${arg}" STREQUAL "xNVCC_OPTIONS")
            set(_hipcc_options_found FALSE)
            set(_hcc_options_found FALSE)
            set(_nvcc_options_found TRUE)
            set(_exclude_platforms_found FALSE)
        elseif("x${arg}" STREQUAL "xEXCLUDE_HIP_PLATFORM")
            set(_hipcc_options_found FALSE)
            set(_hcc_options_found FALSE)
            set(_nvcc_options_found FALSE)
            set(_exclude_platforms_found TRUE)
        else()
            if(_hipcc_options_found)
                list(APPEND ${_hipcc_options} ${arg})
            elseif(_hcc_options_found)
                list(APPEND ${_hcc_options} ${arg})
            elseif(_nvcc_options_found)
                list(APPEND ${_nvcc_options} ${arg})
            elseif(_exclude_platforms_found)
                set(${_exclude_platforms} ${arg})
            else()
                list(APPEND ${_sources} "${_dir}/${arg}")
            endif()
        endif()
    endforeach()
endmacro()

# Helper macro to parse RUN instructions
macro(PARSE_RUN_COMMAND _target _arguments _exclude_platforms)
    set(${_target})
    set(${_arguments} " ")
    set(${_exclude_platforms})
    set(_target_found FALSE)
    set(_exclude_platforms_found FALSE)
    foreach(arg ${ARGN})
        if(NOT _target_found)
            set(_target_found TRUE)
            set(${_target} ${arg})
        elseif("x${arg}" STREQUAL "xEXCLUDE_HIP_PLATFORM")
            set(_exclude_platforms_found TRUE)
        else()
            if(_exclude_platforms_found)
                set(${_exclude_platforms} ${arg})
            else()
                list(APPEND ${_arguments} ${arg})
            endif()
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

# Helper macro to create a test
macro(MAKE_TEST exe)
    string(REPLACE " " "" smush_args ${ARGN})
    set(testname ${PROJECT_NAME}/${exe}${smush_args}.tst)
    add_test(NAME ${testname} COMMAND ${PROJECT_BINARY_DIR}/${exe} ${ARGN})
    set_tests_properties(${testname} PROPERTIES PASS_REGULAR_EXPRESSION "PASSED")
endmacro()
#-------------------------------------------------------------------------------

# Macro: HIT_ADD_FILES used to scan+add multiple files for testing.
macro(HIT_ADD_FILES _dir)
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
            parse_build_command(_target _sources _hipcc_options _hcc_options _nvcc_options _exclude_platforms ${_dir} ${_cmd})
            insert_into_map("_exclude" "${_target}" "${_exclude_platforms}")
            if(_exclude_platforms STREQUAL "all" OR _exclude_platforms STREQUAL ${HIP_PLATFORM})
            else()
                set_source_files_properties(${_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
                hip_add_executable(${_target} ${_sources} HIPCC_OPTIONS ${_hipcc_options} HCC_OPTIONS ${_hcc_options} NVCC_OPTIONS ${_nvcc_options})
            endif()
        endforeach()

        # Add tests
        execute_process(COMMAND ${HIP_SRC_PATH}/tests/hit/parser --runCMDs ${file}
            OUTPUT_VARIABLE _contents
            ERROR_QUIET
            WORKING_DIRECTORY ${_dir}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "\n" ";" _contents "${_contents}")
        foreach(_cmd ${_contents})
            string(REGEX REPLACE " " ";" _cmd "${_cmd}")
            parse_run_command(_target _arguments _exclude_platforms ${_cmd})
            read_from_map("_exclude" "${_target}" _exclude_platforms_from_build)
            if(_exclude_platforms STREQUAL "all" OR _exclude_platforms STREQUAL ${HIP_PLATFORM} OR
                    _exclude_platforms_from_build STREQUAL "all" OR _exclude_platforms_from_build STREQUAL ${HIP_PLATFORM})
            else()
                make_test(${_target} ${_arguments})
            endif()
        endforeach()
    endforeach()
endmacro()

# Macro: HIT_ADD_DIRECTORY to scan+add all files in a directory for testing
macro(HIT_ADD_DIRECTORY _dir)
    file(GLOB files "${_dir}/*.c*")
    hit_add_files(${_dir} ${files})
endmacro()

# Macro: HIT_ADD_DIRECTORY_RECURSIVE to scan+add all files in a directory+subdirectories for testing
macro(HIT_ADD_DIRECTORY_RECURSIVE _dir)
    file(GLOB children RELATIVE ${_dir} ${_dir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${_dir}/${child})
            hit_add_directory_recursive(${_dir}/${child})
        else()
            hit_add_files(${_dir} ${child})
        endif()
    endforeach()
endmacro()

# vim: ts=4:sw=4:expandtab:smartindent
