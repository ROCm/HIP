set(TEST_SRC
    gl_interop_common.cc
    hipGLGetDevices.cc
    hipGraphicsMapResources.cc
)

hip_add_exe_to_target(NAME GLInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
target_link_libraries(GLInteropTest OpenGL::GL OpenGL::EGL)