#include "../device_tests_common.hh"
// this file is named minf.cc to not duplicate CMake target names
GENERATE_KERNEL_FLOAT(min, min(1.0f, 1.0f));