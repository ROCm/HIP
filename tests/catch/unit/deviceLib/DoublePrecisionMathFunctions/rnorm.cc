#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(rnorm, rnorm(1, const_cast<double*>(a)));