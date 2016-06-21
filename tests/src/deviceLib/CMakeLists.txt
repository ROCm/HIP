cmake_minimum_required (VERSION 2.6)

project (deviceLib)

include_directories( ${HIPTEST_SOURCE_DIR} )

build_hip_executable (hip_ballot hip_ballot.cpp)
make_test(hip_ballot " " )

build_hip_executable (hip_anyall hip_anyall.cpp)
make_test(hip_anyall " " )

build_hip_executable (hip_popc hip_popc.cpp)
make_test(hip_popc " " )

build_hip_executable (hip_clz hip_clz.cpp)
make_test(hip_clz " " )

build_hip_executable (hip_brev hip_brev.cpp)
make_test(hip_brev " " )

build_hip_executable (hip_ffs hip_ffs.cpp)
make_test(hip_ffs " " )

build_hip_executable_sm35 (hip_test_ldg  hip_test_ldg.cpp)
make_test(hip_test_ldg " " )


build_hip_executable (hipSimpleAtomicsTest hipSimpleAtomicsTest.cpp)
make_test(hipSimpleAtomicsTest " ")

build_hip_executable (hipMathFunctionsHost hipMathFunctions.cpp hipSinglePrecisionMathHost.cpp hipDoublePrecisionMathHost.cpp)
make_test(hipMathFunctionsHost " ")

build_hip_executable (hipMathFunctionsDevice hipMathFunctions.cpp hipSinglePrecisionMathDevice.cpp hipDoublePrecisionMathDevice.cpp)
make_test(hipMathFunctionsDevice " ")

build_hip_executable (hipIntrinsics hipMathFunctions.cpp hipSinglePrecisionIntrinsics.cpp hipDoublePrecisionIntrinsics.cpp hipIntegerIntrinsics.cpp)
make_test(hipIntrinsics " ")

build_hip_executable (hipTestDevice hipTestDevice.cpp)
make_test(hipTestDevice " ")

build_hip_executable (hipTestDeviceDouble hipTestDeviceDouble.cpp)
make_test(hipTestDeviceDouble " ")

