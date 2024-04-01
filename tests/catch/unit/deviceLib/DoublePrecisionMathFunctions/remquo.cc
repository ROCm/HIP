#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(remquo, remquo(1.0, 2.0, reinterpret_cast<int*>(a)));
