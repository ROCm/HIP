#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicAdd_int, int, atomicAdd((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_usigned_int, unsigned int, atomicAdd((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_unsigned_long_long, unsigned long long, atomicAdd((unsigned long long*)a, (unsigned long long)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_float, float, atomicAdd((float*)a, (float)1))
GENERATE_KERNEL_ATOMICS(atomicAdd_double, double, atomicAdd((double*)a, (double)1))
