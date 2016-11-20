#include <hip/hip_runtime_api.h>
#include "test_common.h"

int main()
{
    hipLimit_t lim = hipLimitMallocHeapSize;
    HIP_PRINT_STATUS(hipDeviceGetLimit(NULL, lim));
}
