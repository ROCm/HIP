#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include"hipDeviceUtil.h"

int main()
{
    int deviceCnt;
    HIP_CHECK(hipGetDeviceCount(&deviceCnt), hipGetDeviceCount);
    HIP_CHECK(hipGetDeviceCount(0), hipGetDeviceCount);
}
