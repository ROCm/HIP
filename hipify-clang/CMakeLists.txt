cmake_minimum_required(VERSION 2.8.8)
project(hipify-clang)

set(BUILD_HIPIFY_CLANG 0 CACHE INTERNAL "")

if (HIPIFY_CLANG_LLVM_DIR)
    find_package(LLVM PATHS ${HIPIFY_CLANG_LLVM_DIR} REQUIRED NO_DEFAULT_PATH)
else()
    message(STATUS "hipify-clang will not be built. To build it please specify absolute path to LLVM 3.8 or higher using HIPIFY_CLANG_LLVM_DIR")
    return()
endif()

option(HIPIFY_CLANG_TESTS "Build the tests for hipify-clang, if lit is installed" ON)

# Disable the tests if `lit` is not installed.
find_program(LIT_COMMAND lit)
if (NOT LIT_COMMAND)
    set(HIPIFY_CLANG_TESTS OFF CACHE INTERNAL "")
    message(STATUS "hipify-clang's tests are not being built because `lit` is not installed.")
endif()

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_llvm_executable(hipify-clang src/Cuda2Hip.cpp)

set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)

# Link against LLVM and CLANG libraries
target_link_libraries(hipify-clang
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

if(WIN32)
    target_link_libraries(hipify-clang version)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS}")
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR- /EHs- /EHc-")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /SUBSYSTEM:WINDOWS")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fno-rtti -fvisibility-inlines-hidden")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIPIFY_CLANG_RES=\\\"${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}\\\"")

install(TARGETS hipify-clang DESTINATION bin)

if (HIPIFY_CLANG_TESTS)
    # tests
    find_package(PythonInterp 2.7 REQUIRED EXACT)

    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/../tests/hipify-clang/lit.site.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/tests/hipify-clang/lit.site.cfg
        @ONLY
    )

    add_lit_testsuite(test-hipify "Running HIPify regression tests"
        ${CMAKE_CURRENT_LIST_DIR}/../tests/hipify-clang
        PARAMS site_config=${CMAKE_CURRENT_BINARY_DIR}/tests/hipify-clang/lit.site.cfg
        DEPENDS hipify-clang lit
    )

    add_custom_target(test-hipify-clang)
    add_dependencies(test-hipify-clang test-hipify)
    set_target_properties(test-hipify-clang PROPERTIES FOLDER "Tests")
endif()

set(BUILD_HIPIFY_CLANG 1 CACHE INTERNAL "")
