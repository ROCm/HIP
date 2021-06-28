# Common Tests - Test independent of all platforms
set(TEST_SRC
    memset.cc
    malloc.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArrayAsync.cc
    hipMemcpyPeer.cc
    hipMemcpyPeerAsync.cc
    hipMemcpy3D.cc
    hipMemcpy3DAsync.cc
    hipMemcpyParam2D.cc
    hipMemcpyParam2DAsync.cc
    hipMemcpy2D.cc
    hipMemcpy2DAsync.cc
)

# Create shared lib of all tests
add_library(MemoryTest SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests MemoryTest)
