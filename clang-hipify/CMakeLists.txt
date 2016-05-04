cmake_minimum_required(VERSION 2.8.8)
project(hipify-clang)

# Find LLVM package
find_package(LLVM 3.8 QUIET PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
if (NOT ${LLVM_FOUND})
    message(FATAL_ERROR "Don't know where to find LLVM (v3.8) package. Please specify absolute path using -DLLVM_DIR")
endif()

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_llvm_executable(hipify-clang src/Cuda2Hip.cpp )
find_program(LIT_COMMAND lit)

# Link against LLVM and CLANG tools libraries
target_link_libraries(hipify-clang
  clangASTMatchers
  clangFrontend
  clangTooling
  clangParse
  clangSerialization
  clangSema
  clangEdit
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

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fno-rtti -fvisibility-inlines-hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIPIFY_CLANG_RES=\\\"${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}\\\"")

install(TARGETS hipify-clang
        DESTINATION bin)

# tests
set(Python_ADDITIONAL_VERSIONS 2.7)
include(FindPythonInterp)
if( NOT PYTHONINTERP_FOUND )
  message(FATAL_ERROR
          "Unable to find Python interpreter, required for builds and testing\n\n"
          "Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
endif()

set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}  )

configure_file(
  ${CMAKE_SOURCE_DIR}/tests/clang-hipify/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/tests/clang-hipify/lit.site.cfg
  @ONLY)

add_lit_testsuite(test-hipify "Running HIPify regression tests"
  ${CMAKE_SOURCE_DIR}/tests/clang-hipify
  PARAMS site_config=${CMAKE_CURRENT_BINARY_DIR}/tests/clang-hipify/lit.site.cfg
  DEPENDS hipify-clang lit
  )

add_custom_target(test-clang-hipify)
add_dependencies(test-clang-hipify test-hipify)
set_target_properties(test-clang-hipify PROPERTIES FOLDER "Tests")
