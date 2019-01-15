if (CUDA_VERSION VERSION_GREATER "9.2")
    cmake_minimum_required(VERSION 3.12.3)
else()
    if (CUDA_VERSION VERSION_LESS "9.0")
        cmake_minimum_required(VERSION 3.5.0)
    else()
        cmake_minimum_required(VERSION 3.6.0)
    endif()
    if (MSVC AND MSVC_VERSION VERSION_GREATER "1900")
        cmake_minimum_required(VERSION 3.7.2)
    endif()
endif()

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

if(WIN32)
    target_link_libraries(hipify-clang PRIVATE version)
endif()

if ((LLVM_PACKAGE_VERSION VERSION_EQUAL "7") OR (LLVM_PACKAGE_VERSION VERSION_GREATER "7"))
    target_link_libraries(hipify-clang PRIVATE clangToolingInclusions)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS}")
if(MSVC)
    target_compile_options(hipify-clang PRIVATE "/Od /GR- /EHs- /EHc-")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /SUBSYSTEM:WINDOWS")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fno-rtti -fvisibility-inlines-hidden")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIPIFY_CLANG_RES=\\\"${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}\\\"")

install(TARGETS hipify-clang DESTINATION bin)

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
         CUDA_VERSION VERSION_EQUAL "10.0")
        message(SEND_ERROR "CUDA ${CUDA_VERSION} is not supported by clang ${LLVM_PACKAGE_VERSION}.")
        if (CUDA_VERSION VERSION_LESS "7.0")
            message(STATUS "Please install CUDA 7.0 or higher.")
        elseif ((CUDA_VERSION VERSION_EQUAL "7.0") OR (CUDA_VERSION VERSION_EQUAL "7.5"))
            message(STATUS "Please install clang 3.8 or higher.")
        elseif (CUDA_VERSION VERSION_EQUAL "8.0")
            message(STATUS "Please install clang 4.0 or higher.")
        elseif (CUDA_VERSION VERSION_EQUAL "9.0")
            message(STATUS "Please install clang 6.0 or higher.")
        elseif ((CUDA_VERSION VERSION_EQUAL "9.1") OR (CUDA_VERSION VERSION_EQUAL "9.2"))
            message(STATUS "Please install clang 7.0 or higher.")
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
