#include <hip/hip_runtime_api.h>
#include "test_common.h"

int main()
{
    HIP_PRINT_STATUS(hipSetDevice(-1));
    int count;
    hipGetDeviceCount(&count);
    HIP_PRINT_STATUS(hipSetDevice(count+1));
}
