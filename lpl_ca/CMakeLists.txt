#-------------------------------------LPL--------------------------------------#
add_executable(lpl lpl.cpp)
set_target_properties(
    lpl PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_include_directories(lpl PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_compile_options(lpl PUBLIC -Wall)
target_link_libraries(lpl PUBLIC pthread)

install(TARGETS lpl RUNTIME DESTINATION bin)
#-------------------------------------LPL--------------------------------------#

#-------------------------------------CA---------------------------------------#
add_executable(ca ca.cpp ${PROJECT_SOURCE_DIR}/src/code_object_bundle.cpp)
set_target_properties(
    ca PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_include_directories(ca SYSTEM PUBLIC ${HSA_PATH}/include)
target_include_directories(ca PUBLIC ${PROJECT_SOURCE_DIR}/src)
find_library(
    hsart NAMES libhsa-runtime64.so libhsa-runtime64.so.1 HINTS ${HSA_PATH}/lib)
target_link_libraries(ca PUBLIC ${hsart})
target_compile_options(ca PUBLIC -Wall)

install(TARGETS ca RUNTIME DESTINATION bin)
#-------------------------------------CA---------------------------------------#