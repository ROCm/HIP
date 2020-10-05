project(module_api)

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
add_executable(launchKernelHcc.hip.out launchKernelHcc.cpp)
add_executable(defaultDriver.hip.out defaultDriver.cpp)

# Generate code object
add_custom_target(
  codeobj
  ALL
  COMMAND  ${HIP_HIPCC_EXECUTABLE} --genco  ../vcpy_kernel.cpp -o vcpy_kernel.code
  COMMENT "codeobj generated"
)

add_dependencies(runKernel.hip.out codeobj)
add_dependencies(launchKernelHcc.hip.out codeobj)
add_dependencies(defaultDriver.hip.out codeobj)

# Link with HIP
target_link_libraries(runKernel.hip.out hip::host)
target_link_libraries(launchKernelHcc.hip.out hip::host)
target_link_libraries(defaultDriver.hip.out hip::host)
