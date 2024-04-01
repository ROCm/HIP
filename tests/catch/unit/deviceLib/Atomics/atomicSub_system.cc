#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicSub_system_int, int, atomicSub_system((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicSub_system_usigned_int, unsigned int, atomicSub_system((unsigned int*)a, (unsigned int)1))