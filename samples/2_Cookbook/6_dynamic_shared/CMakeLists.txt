project(dynamic_shared)

cmake_minimum_required(VERSION 3.10)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# Find hip
find_package(hip)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# Create the excutable
add_executable(dynamic_shared dynamic_shared.cpp)

# Link with HIP
target_link_libraries(dynamic_shared hip::host)
