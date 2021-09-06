# Common Tests - Test independent of all platforms
set(TEST_SRC
    floatMath.cc
    anyAll.cc
    ballot.cc
    clz.cc
    ffs.cc
    funnelshift.cc
    brev.cc
    popc.cc
    ldg.cc
    syncthreadsand.cc
    syncthreadscount.cc
    syncthreadsor.cc
    threadfence_system.cc
)

# AMD only tests
set(AMD_TEST_SRC
    unsafeAtomicAdd.cc
    vectorTypesDevice.cc
    mbcnt.cc
    bitExtract.cc
    bitInsert.cc
    floatTM.cc
)

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    set_source_files_properties(floatTM.cc PROPERTIES COMPILE_FLAGS -std=c++17)
endif()


# Create shared lib of all tests
add_library(UnitDeviceTests SHARED EXCLUDE_FROM_ALL ${TEST_SRC})

if(HIP_PLATFORM MATCHES "nvidia")
    target_compile_options(UnitDeviceTests PUBLIC --Wno-deprecated-declarations)
endif()

# Add dependency on build_tests to build it on this custom target
add_dependencies(build_tests UnitDeviceTests)
