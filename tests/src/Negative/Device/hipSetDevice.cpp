#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include"hipDeviceUtil.h"

int main()
{
    HIP_CHECK(hipSetDevice(0), hipSetDevice);
    HIP_CHECK(hipSetDevice(1026), hipSetDevice);
    HIP_CHECK(hipSetDevice(-1), hipSetDevice);
}
