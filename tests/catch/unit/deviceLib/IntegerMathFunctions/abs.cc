#include "../device_tests_common.hh"

GENERATE_KERNEL_INTEGER(abs_int, abs(1));

GENERATE_KERNEL_INTEGER(abs_longint, abs(2l));

GENERATE_KERNEL_INTEGER(abs_longlongint, abs(3ll));
