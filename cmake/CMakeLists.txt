

#Our FindHCC is in the same directory as the this CMakeLists.txt, which is CMAKE_SOURCE_DIR
# Let cmake know where to find FindHCC.cmake
#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

find_package(HIP REQUIRED)

message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")

