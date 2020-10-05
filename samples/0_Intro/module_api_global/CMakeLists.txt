project(modile_api_global)

cmake_minimum_required(VERSION 3.10)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(hip)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# Create the excutable
add_executable(runKernel.hip.out runKernel.cpp)

# Generate code object
add_custom_target(
  codeobj
  ALL
  COMMAND  ${HIP_HIPCC_EXECUTABLE} --genco  ../vcpy_kernel.cpp -o vcpy_kernel.code
  COMMENT "codeobj generated"
)

add_dependencies(runKernel.hip.out codeobj)

# Link with HIP
target_link_libraries(runKernel.hip.out hip::host)