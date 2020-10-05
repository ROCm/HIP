project(texture2dDrv)

cmake_minimum_required(VERSION 3.10)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(hip)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

# Create the excutable
add_executable(texture2dDrv texture2dDrv.cpp)

# Generate code object
add_custom_target(
  codeobj
  ALL
  COMMAND  ${HIP_HIPCC_EXECUTABLE} --genco  ../tex2dKernel.cpp -o tex2dKernel.code
  COMMENT "codeobj generated"
)

add_dependencies(texture2dDrv codeobj)

# Link with HIP
target_link_libraries(texture2dDrv hip::host)
