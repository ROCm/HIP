#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(frexp, frexp(1.0, reinterpret_cast<int*>(a)));
