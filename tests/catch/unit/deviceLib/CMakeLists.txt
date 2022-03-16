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
    threadfence_system.cc
    hipTestDeviceSymbol.cc
)

# skipped for windows compiler issue - Illegal instruction detected
if(UNIX)
    set(TEST_SRC ${TEST_SRC}
                 syncthreadsand.cc
                 syncthreadscount.cc
                 syncthreadsor.cc)
endif()


# AMD only tests
set(AMD_TEST_SRC
    unsafeAtomicAdd.cc
    vectorTypesDevice.cc
    mbcnt.cc
    bitExtract.cc
    bitInsert.cc
    floatTM.cc
    hipMathFunctions.cc
)

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    set_source_files_properties(floatTM.cc PROPERTIES COMPILE_FLAGS -std=c++17)
    set_source_files_properties(hipMathFunctions.cc PROPERTIES COMPILE_FLAGS "-Xclang -fallow-half-arguments-and-returns")
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS --Wno-deprecated-declarations)
endif()
