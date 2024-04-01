#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicMax_system_int, int, atomicMax_system((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicMax_system_usigned_int, unsigned int, atomicMax_system((unsigned int*)a, (unsigned int)1))