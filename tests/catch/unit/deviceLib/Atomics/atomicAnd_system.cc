#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicAnd_system_int, int, atomicAdd_system((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicAnd_system_unsigned_int, unsigned int, atomicAdd_system((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicAnd_system_unsigned_long_long, unsigned long long, atomicAdd_system((unsigned long long*)a, (unsigned long long)1))
