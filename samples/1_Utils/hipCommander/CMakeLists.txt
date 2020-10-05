project(hipCommander)

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
add_executable(hipCommander hipCommander.cpp)

# Generate code object
add_custom_target(
  codeobj
  ALL
  COMMAND  ${HIP_HIPCC_EXECUTABLE} --genco  ../nullkernel.hip.cpp -o nullkernel.hsaco
  COMMENT "codeobj generated"
)

add_dependencies(hipCommander codeobj)

# Link with HIP
target_link_libraries(hipCommander hip::host)
set_property(TARGET hipCommander PROPERTY CXX_STANDARD 11)
