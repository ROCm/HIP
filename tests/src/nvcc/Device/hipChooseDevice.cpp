#include<hip/hip_runtime_api.h>
#include"test_common.h"

int main()
{
    int dev;
    hipDeviceProp_t prop;
    HIP_PRINT_STATUS(hipChooseDevice(&dev, &prop));
    HIP_PRINT_STATUS(hipChooseDevice(0, &prop));
    HIP_PRINT_STATUS(hipChooseDevice(0, 0));
}
