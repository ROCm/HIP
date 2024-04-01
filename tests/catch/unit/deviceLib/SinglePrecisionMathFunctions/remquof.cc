#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(remquof, remquof(1.0f, 2.0f, reinterpret_cast<int*>(a)));