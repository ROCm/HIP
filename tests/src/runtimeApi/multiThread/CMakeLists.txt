cmake_minimum_required (VERSION 2.6)

project (runtimeApi/multiThread)

include_directories( ${HIPTEST_SOURCE_DIR} )

build_hip_executable (hipMultiThreadStreams1 hipMultiThreadStreams1.cpp)
build_hip_executable (hipMultiThreadStreams2 hipMultiThreadStreams2.cpp) 
build_hip_executable (hipMultiThreadDevice hipMultiThreadDevice.cpp) 

#make_test(hipMultiThreadStreams1 " " )  Fails if 0x3 specified, passes otherwise.  
make_test(hipMultiThreadStreams2 " " )
make_named_test (hipMultiThreadDevice "hipMultiThreadDevice-serial" --tests 0x1) 
make_named_test (hipMultiThreadDevice "hipMultiThreadDevice-pyramid" --tests 0x4) 
make_named_test (hipMultiThreadDevice "hipMultiThreadDevice-nearzero" --tests 0x10) 
