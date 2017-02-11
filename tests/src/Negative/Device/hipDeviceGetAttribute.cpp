#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include"hipDeviceUtil.h"

int main()
{
    int pi;
    int attr = 0;
//    hipDeviceAttribute_t attr = hipDeviceAttributeMaxThreadsPerBlock;
    HIP_CHECK(hipDeviceGetAttribute(NULL, hipDeviceAttribute_t(attr), 0), hipDeviceGetAttribute);
    HIP_CHECK(hipDeviceGetAttribute(&pi, hipDeviceAttribute_t(attr), 0), hipDeviceGetAttribute);
    attr = -1;
    HIP_CHECK(hipDeviceGetAttribute(NULL, hipDeviceAttribute_t(attr), 0), hipDeviceGetAttribute);
    HIP_CHECK(hipDeviceGetAttribute(&pi, hipDeviceAttribute_t(attr), 0), hipDeviceGetAttribute);
    attr = 0;
    HIP_CHECK(hipDeviceGetAttribute(NULL, hipDeviceAttribute_t(attr), -1), hipDeviceGetAttribute);
    HIP_CHECK(hipDeviceGetAttribute(&pi, hipDeviceAttribute_t(attr), -1), hipDeviceGetAttribute);
    attr = -1;
    HIP_CHECK(hipDeviceGetAttribute(NULL, hipDeviceAttribute_t(attr), -1), hipDeviceGetAttribute);
    HIP_CHECK(hipDeviceGetAttribute(&pi, hipDeviceAttribute_t(attr), -1), hipDeviceGetAttribute);
}
