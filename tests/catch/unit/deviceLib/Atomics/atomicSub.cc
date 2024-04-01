#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicSub_int, int, atomicSub((int*)a, (int)1))
GENERATE_KERNEL_ATOMICS(atomicSub_usigned_int, unsigned int, atomicSub((unsigned int*)a, (unsigned int)1))