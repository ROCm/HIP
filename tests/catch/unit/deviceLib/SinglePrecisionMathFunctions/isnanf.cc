#include "../device_tests_common.hh"
// this file is named isnanf.cc to not duplicate CMake target names
GENERATE_KERNEL_FLOAT(isnan, isnan(1.0f));