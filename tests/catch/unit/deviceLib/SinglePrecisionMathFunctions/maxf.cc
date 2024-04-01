#include "../device_tests_common.hh"
// this file is named maxf.cc to not duplicate CMake target names
GENERATE_KERNEL_FLOAT(max, max(1.0f, 1.0f));