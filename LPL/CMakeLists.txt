add_executable(lpl lpl.cpp)
set_target_properties(
    lpl PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_include_directories(lpl PUBLIC ${PROJECT_SOURCE_DIR}/src)
# Install LPL if platform is hcc
if (HIP_PLATFORM STREQUAL "hcc")
    install(TARGETS lpl RUNTIME DESTINATION bin)
endif ()