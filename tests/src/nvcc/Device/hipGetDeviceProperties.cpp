#include <hip/hip_runtime_api.h>
#include "test_common.h"

int main()
{
    hipDeviceProp_t prop;
    HIP_PRINT_STATUS(hipGetDeviceProperties(&prop, -1));
    int cnt;
    hipGetDeviceCount(&cnt);
    HIP_PRINT_STATUS(hipGetDeviceProperties(&prop, cnt+1));
    HIP_PRINT_STATUS(hipGetDeviceProperties(NULL, 0));
}
