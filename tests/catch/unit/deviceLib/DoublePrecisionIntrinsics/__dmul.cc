#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(__dmul_rd, __dmul_rd(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__dmul_rn, __dmul_rn(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__dmul_ru, __dmul_ru(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__dmul_rz, __dmul_rz(1.0, 1.0));