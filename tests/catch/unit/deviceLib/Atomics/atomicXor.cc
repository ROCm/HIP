#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicXor_int, int, atomicExch((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicXor_usigned_int, unsigned int, atomicExch((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicXor_unsigned_long_long, unsigned long long, atomicExch((unsigned long long*)a, (unsigned long long)1))