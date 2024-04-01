#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicInc_unsigned_int, unsigned int, atomicInc((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicInc_unsigned_int_two_args, unsigned int, atomicInc((unsigned int*)a, (unsigned int)1))
