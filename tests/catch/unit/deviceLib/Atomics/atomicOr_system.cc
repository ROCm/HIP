#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicOr_system_int, int, atomicExch_system((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicOr_system_usigned_int, unsigned int, atomicExch_system((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicOr_system_unsigned_long_long, unsigned long long, atomicExch_system((unsigned long long*)a, (unsigned long long)1))
GENERATE_KERNEL_ATOMICS(atomicOr_system_float, float, atomicExch_system((float*)a, (float)1))