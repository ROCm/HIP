set(TEST_SRC
    hipGLGetDevices.cc
    hipGraphicsResourceGetMappedPointer.cc
)

hip_add_exe_to_target(NAME GLInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(GLUT REQUIRED)
target_link_libraries(GLInteropTest OpenGL::GL OpenGL::EGL GLUT::GLUT)