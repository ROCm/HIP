#include <hip/hip_runtime_api.h>
#include "test_common.h"

int main()
{
    HIP_PRINT_STATUS(hipGetDeviceCount(NULL));
}
