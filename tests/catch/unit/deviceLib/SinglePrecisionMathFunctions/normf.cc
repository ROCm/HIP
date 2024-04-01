#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(normf, normf(1, const_cast<float*>(a)));