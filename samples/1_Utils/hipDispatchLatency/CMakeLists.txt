project(hipDispatchLatency)

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
add_executable(hipDispatchLatency hipDispatchLatency.cpp)
add_executable(hipDispatchEnqueueRateMT hipDispatchEnqueueRateMT.cpp)

# Generate code object
add_custom_target(
  codeobj
  ALL
  COMMAND  ${HIP_HIPCC_EXECUTABLE} --genco  ../test_kernel.cpp -o test_kernel.code
  COMMENT "codeobj generated"
)

add_dependencies(hipDispatchLatency codeobj)
add_dependencies(hipDispatchEnqueueRateMT codeobj)

# Link with HIP
target_link_libraries(hipDispatchLatency hip::host)
target_link_libraries(hipDispatchEnqueueRateMT hip::host)
set_property(TARGET hipDispatchLatency PROPERTY CXX_STANDARD 11)
set_property(TARGET hipDispatchEnqueueRateMT PROPERTY CXX_STANDARD 11)
