cmake_minimum_required(VERSION 3.5.1)

project(hipify-clang)

if (MSVC AND MSVC_VERSION VERSION_LESS "1900")
    message(SEND_ERROR "hipify-clang could be built by Visual Studio 14 2015 or higher.")
    return()
endif()

find_package(LLVM REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}:")
message(STATUS "   - CMake module path: ${LLVM_CMAKE_DIR}")
message(STATUS "   - Include path     : ${LLVM_INCLUDE_DIRS}")
message(STATUS "   - Binary path      : ${LLVM_TOOLS_BINARY_DIR}")

option(HIPIFY_CLANG_TESTS "Build the tests for hipify-clang, if lit is installed" OFF)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

file(GLOB_RECURSE HIPIFY_SOURCES src/*.cpp)
file(GLOB_RECURSE HIPIFY_HEADERS src/*.h)
add_llvm_executable(hipify-clang ${HIPIFY_SOURCES} ${HIPIFY_HEADERS})

set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)

# Link against LLVM and CLANG libraries
target_link_libraries(hipify-clang PRIVATE
    clangASTMatchers
    clangFrontend
    clangTooling
    clangParse
    clangSerialization
    clangSema
    clangEdit
    clangFormat
    clangLex
    clangAnalysis
    clangDriver
    clangAST
    clangToolingCore
    clangRewrite
    clangBasic
    LLVMProfileData
    LLVMSupport
    LLVMMCParser
    LLVMMC
    LLVMBitReader
    LLVMOption
    LLVMCore)

if (LLVM_PACKAGE_VERSION VERSION_GREATER "6.0.1")
    target_link_libraries(hipify-clang PRIVATE clangToolingInclusions)
endif()

if (LLVM_PACKAGE_VERSION VERSION_GREATER "9.0.1")
    target_link_libraries(hipify-clang PRIVATE LLVMFrontendOpenMP)
endif()

if (MSVC)
    target_link_libraries(hipify-clang PRIVATE version)
    target_compile_options(hipify-clang PRIVATE "/Od /GR- /EHs- /EHc-")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /SUBSYSTEM:WINDOWS")
    set(StdCpp "/std:c++")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fno-rtti -fvisibility-inlines-hidden")
    set(StdCpp "-std=c++")
endif()

if (LLVM_PACKAGE_VERSION VERSION_GREATER "9.0")
    string(APPEND StdCpp "14")
# MSVC starting from 1900 (VS 2015) supports only the following c++ std values: c++14|c++17|c++latest
elseif (MSVC)
    set(StdCpp "")
else()
    string(APPEND StdCpp "11")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS} ${StdCpp} -DHIPIFY_CLANG_RES=\\\"${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}\\\"")

set(INSTALL_PATH_DOC_STRING "Installation path for hipify-clang")
set(HIPIFY_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(HIPIFY_INSTALL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/bin" CACHE PATH ${INSTALL_PATH_DOC_STRING} FORCE)
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        if (BIN_INSTALL_DIR)
            set(HIPIFY_INSTALL_PATH "${BIN_INSTALL_DIR}" CACHE PATH ${INSTALL_PATH_DOC_STRING} FORCE)
        else()
            set(HIPIFY_INSTALL_PATH "${PROJECT_BINARY_DIR}/bin" CACHE PATH ${INSTALL_PATH_DOC_STRING} FORCE)
        endif()
    else()
        message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE specified. Valid values are Debug and Release")
    endif()
elseif(BIN_INSTALL_DIR)
    set(HIPIFY_INSTALL_PATH "${BIN_INSTALL_DIR}" CACHE PATH ${INSTALL_PATH_DOC_STRING} FORCE)
endif()

install(TARGETS hipify-clang DESTINATION ${HIPIFY_INSTALL_PATH})

install(
  DIRECTORY ${LLVM_DIR}/../../clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/
  DESTINATION ${HIPIFY_INSTALL_PATH}
  COMPONENT clang-resource-headers
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.modulemap"
  PATTERN "algorithm"
  PATTERN "complex"
  PATTERN "new"
  PATTERN "ppc_wrappers" EXCLUDE
  PATTERN "openmp_wrappers" EXCLUDE
  )

# Package: hipify-clang
if (UNIX)
    set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hipify-clang)
    configure_file(packaging/hipify-clang.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
    add_custom_target(pkg_hipify-clang  COMMAND ${CMAKE_COMMAND} .
        COMMAND rm -rf *.deb *.rpm *.tar.gz
        COMMAND make package
        COMMAND cp *.deb ${PROJECT_BINARY_DIR}
        COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
        COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
        WORKING_DIRECTORY ${BUILD_DIR})

    file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/fixnames
        CONTENT "pwd; for i in *.deb; do mv \"\$i\" \"\${i/.deb/-amd64.deb}\" ; done
        for i in *.rpm ; do mv \$i \${i/.rpm/.x86_64.rpm} ; done
        ")
    add_custom_target(package_hipify-clang
        COMMAND bash ${PROJECT_BINARY_DIR}/fixnames
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        DEPENDS pkg_hipify-clang)
endif()

if (HIPIFY_CLANG_TESTS)
    find_package(PythonInterp 2.7 REQUIRED)

    function (require_program PROGRAM_NAME)
        find_program(FOUND_${PROGRAM_NAME} ${PROGRAM_NAME})
        if (FOUND_${PROGRAM_NAME})
            message(STATUS "Found ${PROGRAM_NAME}: ${FOUND_${PROGRAM_NAME}}")
        else()
            message(SEND_ERROR "Can't find ${PROGRAM_NAME}. Either set HIPIFY_CLANG_TESTS to OFF to disable hipify tests, or install the missing program.")
        endif()
    endfunction()

    require_program(lit)
    require_program(FileCheck)

    # Populates CUDA_TOOLKIT_ROOT_DIR, which is then applied to the lit config to give the
    # value of --cuda-path for the test runs.
    find_package(CUDA REQUIRED)
    if ((CUDA_VERSION VERSION_LESS "7.0") OR (LLVM_PACKAGE_VERSION VERSION_LESS "3.8") OR
        (CUDA_VERSION VERSION_GREATER "7.5" AND LLVM_PACKAGE_VERSION VERSION_LESS "4.0") OR
        (CUDA_VERSION VERSION_GREATER "8.0" AND LLVM_PACKAGE_VERSION VERSION_LESS "6.0") OR
        (CUDA_VERSION VERSION_GREATER "9.0" AND LLVM_PACKAGE_VERSION VERSION_LESS "7.0") OR
        (CUDA_VERSION VERSION_GREATER "9.2" AND LLVM_PACKAGE_VERSION VERSION_LESS "8.0") OR
        (CUDA_VERSION VERSION_GREATER "10.0" AND LLVM_PACKAGE_VERSION VERSION_LESS "9.0"))
        message(SEND_ERROR "CUDA ${CUDA_VERSION} is not supported by clang ${LLVM_PACKAGE_VERSION}.")
        if (CUDA_VERSION_MAJOR VERSION_LESS "7")
            message(STATUS "Please install CUDA 7.0 or higher.")
        elseif (CUDA_VERSION_MAJOR VERSION_LESS "8")
            message(STATUS "Please install clang 3.8 or higher.")
        elseif (CUDA_VERSION_MAJOR VERSION_LESS "9")
            message(STATUS "Please install clang 4.0 or higher.")
        elseif (CUDA_VERSION VERSION_EQUAL "9.0")
            message(STATUS "Please install clang 6.0 or higher.")
        elseif (CUDA_VERSION_MAJOR VERSION_LESS "10")
            message(STATUS "Please install clang 7.0 or higher.")
        elseif (CUDA_VERSION VERSION_EQUAL "10.0")
            message(STATUS "Please install clang 8.0 or higher.")
        elseif (CUDA_VERSION VERSION_EQUAL "10.1")
            message(STATUS "Please install clang 9.0 or higher.")
        endif()
    endif()

    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/../tests/hipify-clang/lit.site.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/tests/hipify-clang/lit.site.cfg
        @ONLY
    )

    add_lit_testsuite(test-hipify "Running HIPify regression tests"
        ${CMAKE_CURRENT_LIST_DIR}/../tests/hipify-clang
        PARAMS site_config=${CMAKE_CURRENT_BINARY_DIR}/tests/hipify-clang/lit.site.cfg
        ARGS -v
        DEPENDS hipify-clang
    )

    add_custom_target(test-hipify-clang)
    add_dependencies(test-hipify-clang test-hipify)
    set_target_properties(test-hipify-clang PROPERTIES FOLDER "Tests")
endif()
