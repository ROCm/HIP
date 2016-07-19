cmake_minimum_required(VERSION 2.8.3)
project(hip)

#############################
# Configure variables
#############################
# Determine HIP_VERSION
execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/hipconfig --version
                OUTPUT_VARIABLE HIP_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "." ";" VERSION_LIST ${HIP_VERSION})
list(GET VERSION_LIST 0 HIP_VERSION_MAJOR)
list(GET VERSION_LIST 1 HIP_VERSION_MINOR)
list(GET VERSION_LIST 2 HIP_VERSION_PATCH)


# Determine HIP_PLATFORM
if(NOT DEFINED HIP_PLATFORM)
    if(NOT DEFINED ENV{HIP_PLATFORM})
        execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/hipconfig --platform
                        OUTPUT_VARIABLE HIP_PLATFORM
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        set(HIP_PLATFORM $ENV{HIP_PLATFORM} CACHE STRING "HIP Platform")
    endif()
endif()
message(STATUS "HIP Platform: " ${HIP_PLATFORM})

# If HIP_PLATFORM is hcc, we need HCC_HOME and HSA_PATH to be defined
if(HIP_PLATFORM STREQUAL "hcc")
    # Determine HCC_HOME
    if(NOT DEFINED HCC_HOME)
        if(NOT DEFINED ENV{HCC_HOME})
            set(HCC_HOME "/opt/rocm/hcc" CACHE PATH "Path to which HCC has been installed")
        else()
            set(HCC_HOME $ENV{HCC_HOME} CACHE PATH "Path to which HCC has been installed")
        endif()
    endif()
    if(IS_ABSOLUTE ${HCC_HOME} AND EXISTS ${HCC_HOME} AND IS_DIRECTORY ${HCC_HOME})
        execute_process(COMMAND ${HCC_HOME}/bin/hcc --version
                        COMMAND cut -d\  -f9
                        OUTPUT_VARIABLE HCC_VERSION
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "Looking for HCC in: " ${HCC_HOME} ". Found version: " ${HCC_VERSION})
    else()
        message(FATAL_ERROR "Don't know where to find HCC. Please specify abolute path using -DHCC_HOME")
    endif()

    # Determine HSA_PATH
    if(NOT DEFINED HSA_PATH)
        if(NOT DEFINED ENV{HSA_PATH})
            set(HSA_PATH "/opt/rocm/hsa" CACHE PATH "Path to which HSA runtime has been installed")
        else()
            set(HSA_PATH $ENV{HSA_PATH} CACHE PATH "Path to which HSA runtime has been installed")
        endif()
    endif()
    if(IS_ABSOLUTE ${HSA_PATH} AND EXISTS ${HSA_PATH} AND IS_DIRECTORY ${HSA_PATH})
        message(STATUS "Looking for HSA runtime in: " ${HSA_PATH})
    else()
        message(FATAL_ERROR "Don't know where to find HSA runtime. Please specify absolute path using -DHSA_PATH")
    endif()
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Determine HIP install path
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
    if(CMAKE_BUILD_TYPE MATCHES Debug)
            set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Installation path for HIP" FORCE)
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
            set(CMAKE_INSTALL_PREFIX "/opt/rocm/hip" CACHE PATH "Installation path for HIP" FORCE)
    else()
        message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE specified. Valid values are Debug and Release")
    endif()
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    message(STATUS "HIP will be installed in: " ${CMAKE_INSTALL_PREFIX})
else()
    message(FATAL_ERROR "Don't know where to install HIP. Please specify absolute path using -DCMAKE_INSTALL_PREFIX")
endif()

# Set if we need to build shared or static library
if(NOT DEFINED ENV{HIP_USE_SHARED_LIBRARY})
    set(HIP_USE_SHARED_LIBRARY 0)
else()
    set(HIP_USE_SHARED_LIBRARY $ENV{HIP_USE_SHARED_LIBRARY})
endif()

# Check if we need to build clang hipify
if(NOT DEFINED BUILD_CLANG_HIPIFY)
    if(NOT DEFINED ENV{BUILD_CLANG_HIPIFY})
        set(BUILD_CLANG_HIPIFY 0)
    else()
        set(BUILD_CLANG_HIPIFY $ENV{BUILD_CLANG_HIPIFY})
    endif()
endif()

#############################
# Build steps
#############################
# Build clang hipify if enabled
if(BUILD_CLANG_HIPIFY)
    add_subdirectory(clang-hipify)
endif()

# Build hip_hcc if platform is hcc
if(HIP_PLATFORM STREQUAL "hcc")
    include_directories(${PROJECT_SOURCE_DIR}/include)

    set(CMAKE_CXX_COMPILER "${HCC_HOME}/bin/hcc")
    set(CMAKE_C_COMPILER   "${HCC_HOME}/bin/hcc")

    # Set HIP_HCC so we know this is HIP compile, some files are shared with HCC (staging_buffer).
    set(CMAKE_CXX_FLAGS " -hc -I${HCC_HOME}/include -I${HSA_PATH}/include -I/opt/rocm/libhsakmt/include/libhsakmt -stdlib=libc++ -DHIP_HCC")
    set(CMAKE_C_FLAGS   " -hc -I${HCC_HOME}/include -I${HSA_PATH}/include -I/opt/rocm/libhsakmt/include/libhsakmt -stdlib=libc++ -DHIP_HCC")

    set(SOURCE_FILES src/device_util.cpp
                     src/hip_hcc.cpp
                     src/hip_device.cpp
                     src/hip_error.cpp
                     src/hip_event.cpp
                     src/hip_ldg.cpp
                     src/hip_memory.cpp
                     src/hip_peer.cpp
                     src/hip_stream.cpp
                     src/hip_fp16.cpp
                     src/staging_buffer.cpp)

    if(${HIP_USE_SHARED_LIBRARY} EQUAL 1)
        add_library(hip_hcc SHARED ${SOURCE_FILES})
    else()
        #add_library(hip_hcc STATIC ${SOURCE_FILES})
        add_library(hip_hcc OBJECT ${SOURCE_FILES})
    endif()
endif()

# Build doxygen documentation
add_custom_target(doc COMMAND HIP_PATH=${CMAKE_CURRENT_SOURCE_DIR} doxygen ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen-input/doxy.cfg
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs)

#############################
# Install steps
#############################
# Install hip_hcc if platform is hcc
if(HIP_PLATFORM STREQUAL "hcc")
    if(${HIP_USE_SHARED_LIBRARY} EQUAL 1)
        install(TARGETS hip_hcc DESTINATION lib)
    else()
        #install(TARGETS hip_hcc DESTINATION lib)
        install(DIRECTORY ${PROJECT_BINARY_DIR}/CMakeFiles/hip_hcc.dir/src/ DESTINATION lib)
    endif()
endif()

# Install src, bin, include if necessary
execute_process(COMMAND test ${CMAKE_INSTALL_PREFIX} -ef ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE INSTALL_SOURCE)
if(NOT ${INSTALL_SOURCE} EQUAL 0)
    install(DIRECTORY src DESTINATION .)
    install(DIRECTORY bin DESTINATION . USE_SOURCE_PERMISSIONS)
    install(DIRECTORY include DESTINATION .)
endif()

#############################
# Packaging steps
#############################
# Package: hip_base
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hip_base)
configure_file(packaging/hip_base.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
configure_file(packaging/hip_base.postinst ${BUILD_DIR}/postinst @ONLY)
configure_file(packaging/hip_base.prerm ${BUILD_DIR}/prerm @ONLY)
add_custom_target(pkg_hip_base COMMAND ${CMAKE_COMMAND} .
                               COMMAND rm -rf *.deb *.rpm *.tar.gz
                               COMMAND make package
                               COMMAND cp *.deb ${PROJECT_BINARY_DIR}
                               COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
                               COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
                               WORKING_DIRECTORY ${BUILD_DIR})

# Package: hip_hcc
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hip_hcc)
configure_file(packaging/hip_hcc.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
configure_file(packaging/hip_hcc.postinst ${BUILD_DIR}/postinst @ONLY)
configure_file(packaging/hip_hcc.prerm ${BUILD_DIR}/prerm @ONLY)
add_custom_target(pkg_hip_hcc  COMMAND ${CMAKE_COMMAND} .
                               COMMAND rm -rf *.deb *.rpm *.tar.gz
                               COMMAND make package
                               COMMAND cp *.deb ${PROJECT_BINARY_DIR}
                               COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
                               COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
                               WORKING_DIRECTORY ${BUILD_DIR}
                               DEPENDS hip_hcc)

# Package: hip_nvcc
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hip_nvcc)
configure_file(packaging/hip_nvcc.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
add_custom_target(pkg_hip_nvcc COMMAND ${CMAKE_COMMAND} .
                               COMMAND rm -rf *.deb *.rpm *.tar.gz
                               COMMAND make package
                               COMMAND cp *.deb ${PROJECT_BINARY_DIR}
                               COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
                               COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
                               WORKING_DIRECTORY ${BUILD_DIR})

# Package: hip_doc
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hip_doc)
configure_file(packaging/hip_doc.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
add_custom_target(pkg_hip_doc  COMMAND ${CMAKE_COMMAND} .
                               COMMAND rm -rf *.deb *.rpm *.tar.gz
                               COMMAND make package
                               COMMAND cp *.deb ${PROJECT_BINARY_DIR}
                               COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
                               COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
                               WORKING_DIRECTORY ${BUILD_DIR})

# Package: hip_samples
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/hip_samples)
configure_file(packaging/hip_samples.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
add_custom_target(pkg_hip_samples COMMAND ${CMAKE_COMMAND} .
                                  COMMAND rm -rf *.deb *.rpm *.tar.gz
                                  COMMAND make package
                                  COMMAND cp *.deb ${PROJECT_BINARY_DIR}
                                  COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
                                  COMMAND cp *.tar.gz ${PROJECT_BINARY_DIR}
                                  WORKING_DIRECTORY ${BUILD_DIR})

# Package: all
if(POLICY CMP0037)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0037 OLD)
endif()
add_custom_target(package DEPENDS pkg_hip_base pkg_hip_hcc pkg_hip_nvcc pkg_hip_doc pkg_hip_samples)
if(POLICY CMP0037)
    cmake_policy(POP)
endif()

