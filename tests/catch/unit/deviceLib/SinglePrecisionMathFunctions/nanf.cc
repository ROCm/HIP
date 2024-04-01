#include "../device_tests_common.hh"
__device__ const char *tagp;
GENERATE_KERNEL_FLOAT(nanf, nanf(tagp));