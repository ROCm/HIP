project(inline_asm)

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
add_executable(inline_asm inline_asm.cpp)

# Link with HIP
target_link_libraries(inline_asm hip::host)
