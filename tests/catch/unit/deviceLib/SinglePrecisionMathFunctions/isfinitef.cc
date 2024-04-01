#include "../device_tests_common.hh"
// this file is named isfinitef.cc to not duplicate CMake target names
GENERATE_KERNEL_FLOAT(isfinite, isfinite(1.0f));