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
set(AMD_ARCH_SPEC_TEST_SRC
    AtomicAdd_Coherent_withunsafeflag.cc
    AtomicAdd_Coherent_withoutflag.cc
    AtomicAdd_Coherent_withnoUnsafeflag.cc
    AtomicAdd_NonCoherent_withoutflag.cc
    AtomicAdd_NonCoherent_withnoUnsafeflag.cc
    AtomicAdd_NonCoherent_withunsafeflag.cc
)

if(HIP_PLATFORM MATCHES "amd")
    string(FIND ${OFFLOAD_ARCH_STR} "gfx90a" ARCH_CHECK)
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    set_source_files_properties(floatTM.cc PROPERTIES COMPILE_FLAGS -std=c++17)
if(${ARCH_CHECK} GREATER_EQUAL 0)
	set(TEST_SRC ${TEST_SRC} ${AMD_ARCH_SPEC_TEST_SRC})
    set_source_files_properties(AtomicAdd_Coherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_NonCoherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_Coherent_withnoUnsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_NonCoherent_withnoUnsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    file(GLOB AtomicAdd_files *AtomicAdd_*_*.cc)
    set_property(SOURCE ${AtomicAdd_files} PROPERTY COMPILE_FLAGS --save-temps)
    set_source_files_properties(hipMathFunctions.cc PROPERTIES COMPILE_FLAGS "-Xclang -fallow-half-arguments-and-returns")
endif()
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS --Wno-deprecated-declarations)
endif()
