#ifndef HCC_H
#define HCC_H

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
#include "hip/hcc_detail/hcc_acc.h"
#endif

#endif
