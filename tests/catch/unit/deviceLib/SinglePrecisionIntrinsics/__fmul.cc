#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(__fmul_rd, __fmul_rd(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmul_rn, __fmul_rn(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmul_ru, __fmul_ru(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmul_rz, __fmul_rz(1.0f, 1.0f));