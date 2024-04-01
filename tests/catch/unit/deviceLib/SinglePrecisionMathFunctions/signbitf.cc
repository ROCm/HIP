#include "../device_tests_common.hh"
// this file is called signbitf.cc so that there are no CMake target duplications with signbit.cc
// from double precision tests
GENERATE_KERNEL_FLOAT(signbit, signbit(1.0f));