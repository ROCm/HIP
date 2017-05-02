#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include"hipDeviceUtil.h"

int main()
{
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0), hipGetDeviceProperties);
    HIP_CHECK(hipGetDeviceProperties(NULL, 0), hipGetDeviceProperties);
    HIP_CHECK(hipGetDeviceProperties(NULL, -1), hipGetDeviceProperties);
    HIP_CHECK(hipGetDeviceProperties(&props, -1), hipGetDeviceProperties);
    HIP_CHECK(hipGetDeviceProperties(NULL, 1024), hipGetDeviceProperties);
    HIP_CHECK(hipGetDeviceProperties(&props, 1024), hipGetDeviceProperties);
}
