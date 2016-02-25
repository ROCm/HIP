if(${HIPIFY_STANDLONE})
  cmake_minimum_required(VERSION 2.8.8)
  
  project(hipify)

  find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

  list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
  include(AddLLVM)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  include_directories(${LLVM_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  add_llvm_executable( hipify src/Cuda2Hip.cpp )
  find_program(LIT_COMMAND lit)
  
else()
  set(LLVM_LINK_COMPONENTS
    Option
    Support
  )
  add_clang_executable(hipify src/Cuda2Hip.cpp)
endif()


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

if(${HIPIFY_STANDLONE})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_CFLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CFLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -pthread -fno-rtti -fvisibility-inlines-hidden")
  add_subdirectory(test)
else()
  install(TARGETS hipify
          RUNTIME DESTINATION bin
          COMPONENT clang-extras)
endif()
