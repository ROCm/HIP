cmake_minimum_required (VERSION 2.6)

project (runtime_api)

include_directories( ${HIPTEST_SOURCE_DIR} )
build_hip_executable (hipAPIStreamEnable hipAPIStreamEnable.cpp)
build_hip_executable (hipAPIStreamDisable hipAPIStreamDisable.cpp)
build_hip_executable (hipStreamL5 hipStreamL5.cpp)

# TODO - seg fault
#make_test(hipAPIStreamEnable " ")
#make_test(hipAPIStreamDisable " ")
make_test(hipStreamL5 " ")
