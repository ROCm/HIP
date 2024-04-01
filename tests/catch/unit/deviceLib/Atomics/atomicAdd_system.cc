#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicAdd_system_int, int, atomicAdd_system((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_system_usigned_int, unsigned int, atomicAdd_system((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_system_unsigned_long_long, unsigned long long, atomicAdd_system((unsigned long long*)a, (unsigned long long)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_system_float, float, atomicAdd_system((float*)a, (float)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_system_double, double, atomicAdd_system((double*)a, (double)1))

