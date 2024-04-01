#include "../device_tests_common.hh"
GENERATE_KERNEL_ATOMICS(atomicDec_unsigned_int, unsigned int, atomicDec((unsigned int*)a, (unsigned int)1))
GENERATE_KERNEL_ATOMICS(atomicDec_unsigned_int_two_args, unsigned int, atomicDec((unsigned int*)a, (unsigned int)1))
