#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(__fadd_rd, __fadd_rd(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fadd_rn, __fadd_rn(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fadd_ru, __fadd_ru(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fadd_rz, __fadd_rz(1.0f, 1.0f));