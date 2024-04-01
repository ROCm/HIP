#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(__fmaf_rd, __fmaf_rd(1.0f, 1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmaf_rn, __fmaf_rn(1.0f, 1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmaf_ru, __fmaf_ru(1.0f, 1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmaf_rz, __fmaf_rz(1.0f, 1.0f, 1.0f));