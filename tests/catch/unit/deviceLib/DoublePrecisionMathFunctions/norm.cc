#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(norm, norm(1, const_cast<double*>(a)));