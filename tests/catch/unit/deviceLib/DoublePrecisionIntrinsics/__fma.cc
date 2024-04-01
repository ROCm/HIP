#include "../device_tests_common.hh"
GENERATE_KERNEL_DOUBLE(__fma_rd, __fma_rd(1.0, 1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__fma_rn, __fma_rn(1.0, 1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__fma_ru, __fma_ru(1.0, 1.0, 1.0));
GENERATE_KERNEL_DOUBLE(__fma_rz, __fma_rz(1.0, 1.0, 1.0));