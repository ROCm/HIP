#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicMin_int, int, atomicMin((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicMin_usigned_int, unsigned int, atomicMin((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicMin_unsigned_long_long, unsigned long long, atomicMin((unsigned long long*)a, (unsigned long long)1))