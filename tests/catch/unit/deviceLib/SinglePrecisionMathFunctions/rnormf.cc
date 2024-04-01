#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(rnormf, rnormf(1, const_cast<float*>(a)));