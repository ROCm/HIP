#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(__fmaf_ieee_rd, __fmaf_ieee_rd(1.0f, 1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmaf_ieee_rn, __fmaf_ieee_rn(1.0f, 1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmaf_ieee_ru, __fmaf_ieee_ru(1.0f, 1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fmaf_ieee_rz, __fmaf_ieee_rz(1.0f, 1.0f, 1.0f));