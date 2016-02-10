cmake_minimum_required(VERSION 2.8.8)

project(hipify)

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Now build our tools
add_executable(hipify src/Cuda2Hip.cpp)

# Link against LLVM and CLANG tools libraries
target_link_libraries(hipify
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
  LLVMSupport
  LLVMMCParser
  LLVMMC
  LLVMBitReader
  LLVMOption
  LLVMCore)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -pthread -D __STDC_LIMIT_MACROS -D __STDC_CONSTANT_MACROS  -D __STDC_FORMAT_MACROS -fno-rtti -fvisibility-inlines-hidden")
