#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicXor_system_int, int, atomicExch_system((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicXor_system_usigned_int, unsigned int, atomicExch_system((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicXor_system_unsigned_long_long, unsigned long long, atomicExch_system((unsigned long long*)a, (unsigned long long)1))