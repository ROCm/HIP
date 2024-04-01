#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicMax_int, int, atomicMax((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicMax_usigned_int, unsigned int, atomicMax((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicMax_unsigned_long_long, unsigned long long, atomicMax((unsigned long long*)a, (unsigned long long)1))