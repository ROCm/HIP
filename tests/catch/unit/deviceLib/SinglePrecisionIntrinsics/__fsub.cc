#include "../device_tests_common.hh"
GENERATE_KERNEL_FLOAT(__fsub_rd, __fsub_rd(1.0f, 1.0f)); 
GENERATE_KERNEL_FLOAT(__fsub_rn, __fsub_rn(1.0f, 1.0f)); 
GENERATE_KERNEL_FLOAT(__fsub_ru, __fsub_ru(1.0f, 1.0f)); 
GENERATE_KERNEL_FLOAT(__fsub_rz, __fsub_rz(1.0f, 1.0f)); 