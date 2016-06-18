cmake_minimum_required (VERSION 2.6)

project (runtimeApi/memory)

include_directories( ${HIPTEST_SOURCE_DIR} )

build_hip_executable (hipMemset hipMemset.cpp)
make_test(hipMemset " " )
make_test(hipMemset --N 10    --memsetval 0x42 )  # small copy, just 10 bytes.
make_test(hipMemset --N 10013 --memsetval 0x5a )  # oddball size.
make_test(hipMemset --N 256M  --memsetval 0xa6 )  # big copy

build_hip_executable (hipMemcpy_simple hipMemcpy_simple.cpp) 
make_test(hipMemcpy_simple  " " )

build_hip_executable (hipMemcpy hipMemcpy.cpp)
make_named_test(hipMemcpy "hipMemcpy-modes" --tests 0x1 )
make_named_test(hipMemcpy "hipMemcpy-size" --tests 0x6 )
make_named_test(hipMemcpy "hipMemcpy-multithreaded" --tests 0x8 )

build_hip_executable (hipMemcpyAsync hipMemcpyAsync.cpp)
make_named_test(hipMemcpy_simple "hipMemcpyAsync-simple" --async)
#make_test(hipMemcpyAsync  " " )

build_hip_executable (hipMemoryAllocate hipMemoryAllocate.cpp)

build_hip_executable (hipMemcpyAll hipMemcpyAll.cpp)
#make_test(hipMemcpyAll " ")

# Debug synchronization, then enable.
make_test(hipMemoryAllocate " ")
