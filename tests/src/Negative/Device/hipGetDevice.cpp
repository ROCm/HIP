#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include"hipDeviceUtil.h"

int main()
{
    int device;
    HIP_CHECK(hipGetDevice(NULL), hipGetDevice);
    HIP_CHECK(hipGetDevice(&device), hipGetDevice);
}
