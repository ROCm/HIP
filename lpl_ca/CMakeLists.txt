#-------------------------------------LPL--------------------------------------#
# Look for ROCclr which contains elfio
find_package(ROCclr REQUIRED CONFIG
  PATHS
    /opt/rocm
    /opt/rocm/rocclr)

add_executable(lpl lpl.cpp)
set_target_properties(
    lpl PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_include_directories(lpl
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
  PRIVATE
    $<TARGET_PROPERTY:amdrocclr_static,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_options(lpl PUBLIC -Wall)
target_link_libraries(lpl PUBLIC pthread)
set_target_properties(lpl PROPERTIES LINK_FLAGS_RELEASE -s)
add_custom_command(TARGET lpl POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/lpl ${PROJECT_BINARY_DIR}/bin/lpl)
install(TARGETS lpl RUNTIME DESTINATION bin)
#-------------------------------------LPL--------------------------------------#

#-------------------------------------CA---------------------------------------#
add_executable(ca ca.cpp)
set_target_properties(
    ca PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_include_directories(ca SYSTEM PUBLIC ${HSA_PATH}/include)
target_include_directories(ca PUBLIC ${PROJECT_SOURCE_DIR}/src)

find_package(hsa-runtime64 REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/hsa-runtime64)

target_link_libraries(ca PUBLIC hsa-runtime64::hsa-runtime64 )
target_compile_options(ca PUBLIC -DDISABLE_REDUCED_GPU_BLOB_COPY -Wall)
set_target_properties(ca PROPERTIES LINK_FLAGS_RELEASE -s)
add_custom_command(TARGET ca POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/ca ${PROJECT_BINARY_DIR}/bin/ca)

install(TARGETS ca RUNTIME DESTINATION bin)
#-------------------------------------CA---------------------------------------#
