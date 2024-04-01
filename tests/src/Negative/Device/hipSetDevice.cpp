#include "test_common.h"

int main() {
    int numDevices = 0;

    HIPCHECK_API(hipGetDeviceCount(&numDevices), hipSuccess);
    if (numDevices > 0) {
        for (int deviceId = 0; deviceId < numDevices; deviceId++) {
            HIPCHECK_API(hipSetDevice(deviceId), hipSuccess);
        }
        HIPCHECK_API(hipSetDevice(numDevices), hipErrorInvalidDevice);
        HIPCHECK_API(hipSetDevice(-1), hipErrorInvalidDevice);
    }
    else {
        failed("Error: failed to find any compatible devices.");
    }

    passed();
}
