#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(__fdiv_rd, __fdiv_rd(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fdiv_rn, __fdiv_rn(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fdiv_ru, __fdiv_ru(1.0f, 1.0f));
GENERATE_KERNEL_FLOAT(__fdiv_rz, __fdiv_rz(1.0f, 1.0f));