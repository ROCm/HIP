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
)
set(AMD_ARCH_SPEC_TEST_SRC
    AtomicAdd_withunsafeflag.cc
    AtomicAdd_withoutflag.cc
    AtomicAdd_withnounsafeflag.cc
    FlatAtomicAdd_withunsafeflag.cc
    BuiltIns_fmax.cc
    BuiltIns_fmin.cc
    BuiltIns_fadd.cc
    unsafeAtomicAdd_RTC.cc
)

if(HIP_PLATFORM MATCHES "amd")
    if (DEFINED OFFLOAD_ARCH_STR)
      string(FIND ${OFFLOAD_ARCH_STR} "gfx90a" ARCH_CHECK)
    elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
      string(FIND $ENV{HCC_AMDGPU_TARGET} "gfx90a" ARCH_CHECK)
    else()
      set(ARCH_CHECK -1)
    endif()
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    set_source_files_properties(floatTM.cc PROPERTIES COMPILE_FLAGS -std=c++17)
if(${ARCH_CHECK} GREATER_EQUAL 0)
	set(TEST_SRC ${TEST_SRC} ${AMD_ARCH_SPEC_TEST_SRC})
    file(GLOB NoFlag_files *withnounsafeflag.cc)
    set_property(SOURCE ${NoFlag_files} PROPERTY COMPILE_FLAGS "-mno-unsafe-fp-atomics --save-temps -std=c++17")
    file(GLOB Flag_files *withunsafeflag.cc)
    set_property(SOURCE ${Flag_files} PROPERTY COMPILE_FLAGS "-munsafe-fp-atomics --save-temps -std=c++17")
    file(GLOB WithoutFlag_files *withoutflag.cc)
    set_property(SOURCE ${WithoutFlag_files} PROPERTY COMPILE_FLAGS "--save-temps -std=c++17")
endif()
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS --Wno-deprecated-declarations)
endif()
