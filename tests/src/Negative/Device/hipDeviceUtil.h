#ifndef HIPDEVICEUTIL_H
#define HIPDEVICEUTIL_H

#include "hip/hip_runtime_api.h"
#include <iostream>

#define HIP_CHECK(status, func)                                                                    \
    std::cout << #func << " returned " << hipGetErrorString(status) << " in " << __func__          \
              << " at " << __LINE__ << " in file " << __FILE__ << std::endl;

#endif
