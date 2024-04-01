#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(__ddiv_rd, __ddiv_rd(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__ddiv_rn, __ddiv_rn(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__ddiv_ru, __ddiv_ru(1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__ddiv_rz, __ddiv_rz(1.0, 1.0));