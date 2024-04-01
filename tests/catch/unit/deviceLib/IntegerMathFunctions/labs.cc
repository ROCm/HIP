#include "../device_tests_common.hh"

GENERATE_KERNEL_INTEGER(labs_longint, labs(1l));

GENERATE_KERNEL_INTEGER(labs_longlongint, labs(1ll));