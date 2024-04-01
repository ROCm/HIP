#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(__dadd_rd, __dadd_rd(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__dadd_rn, __dadd_rn(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__dadd_ru, __dadd_ru(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__dadd_rz, __dadd_rz(1.0, 1.0));