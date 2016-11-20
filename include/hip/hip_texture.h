#ifndef HIP_TEXTURE_H
#define HIP_TEXTURE_H

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
#include <hip/hcc_detail/hip_texture.h>
#elif defined(__HIP_PLATFORM_NVCC__) && !defined (__HIP_PLATFORM_HCC__)
#include <hip/nvcc_detail/hip_texture.h>
#else 
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif 


#endif
