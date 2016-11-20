#include <hip/hip_runtime_api.h>
#include "test_common.h"

int main()
{
    HIP_PRINT_STATUS(hipSetDeviceFlags(-1));
    HIP_PRINT_STATUS(hipSetDeviceFlags(11));
}
