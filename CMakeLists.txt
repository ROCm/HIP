cmake_minimum_required(VERSION 2.6)
project(hip_hcc)

set (HCC_HOME $ENV{HCC_HOME})
if (NOT DEFINED HCC_HOME)
    set(HCC_HOME "/opt/hcc")
endif()


MESSAGE ("HCC_HOME=" ${HCC_HOME})

set(HSA_PATH "/opt/hsa")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/opt/hip" CACHE PATH "Default installation path of hip" FORCE)
endif ()

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_COMPILER "${HCC_HOME}/bin/hcc")
set(CMAKE_C_COMPILER "${HCC_HOME}/bin/hcc")

set(CMAKE_C_FLAGS " -hc -I${HCC_HOME}/include -I${HSA_PATH}/include -I${HIP_PATH}/include -stdlib=libc++ ")
set(CMAKE_CXX_FLAGS " -hc -I${HCC_HOME}/include -I${HSA_PATH}/include -I${HIP_PATH}/include -stdlib=libc++ ")

set(SOURCE_FILES src/hip_hcc.cpp src/staging_buffer.cpp)

add_library(hip_hcc STATIC ${SOURCE_FILES})

install(TARGETS hip_hcc DESTINATION lib)
install(DIRECTORY src DESTINATION .)
install(DIRECTORY bin DESTINATION . USE_SOURCE_PERMISSIONS)
install(DIRECTORY include DESTINATION .)


