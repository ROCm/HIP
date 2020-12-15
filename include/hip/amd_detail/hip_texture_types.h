/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  amd_detail/hip_texture_types.h
 *  @brief Defines the different newt vector types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_TEXTURE_TYPES_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_TEXTURE_TYPES_H

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#include <limits.h>
//#include <hip/amd_detail/driver_types.h>
#include <hip/amd_detail/channel_descriptor.h>
#include <hip/amd_detail/texture_types.h>

#if __cplusplus

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#if __HIP__
#define __HIP_TEXTURE_ATTRIB __attribute__((device_builtin_texture_type))
#else
#define __HIP_TEXTURE_ATTRIB
#endif

typedef textureReference* hipTexRef;

template <class T, int texType = hipTextureType1D,
          enum hipTextureReadMode mode = hipReadModeElementType>
struct __HIP_TEXTURE_ATTRIB texture : public textureReference {
    texture(int norm = 0, enum hipTextureFilterMode fMode = hipFilterModePoint,
            enum hipTextureAddressMode aMode = hipAddressModeClamp) {
        normalized = norm;
        readMode = mode;
        filterMode = fMode;
        addressMode[0] = aMode;
        addressMode[1] = aMode;
        addressMode[2] = aMode;
        channelDesc = hipCreateChannelDesc<T>();
        sRGB = 0;
        textureObject = nullptr;
        maxAnisotropy = 0;
        mipmapLevelBias = 0;
        minMipmapLevelClamp = 0;
        maxMipmapLevelClamp = 0;
    }

    texture(int norm, enum hipTextureFilterMode fMode, enum hipTextureAddressMode aMode,
            struct hipChannelFormatDesc desc) {
        normalized = norm;
        readMode = mode;
        filterMode = fMode;
        addressMode[0] = aMode;
        addressMode[1] = aMode;
        addressMode[2] = aMode;
        channelDesc = desc;
        sRGB = 0;
        textureObject = nullptr;
        maxAnisotropy = 0;
        mipmapLevelBias = 0;
        minMipmapLevelClamp = 0;
        maxMipmapLevelClamp = 0;
    }
};

#endif /* __cplusplus */

#endif /* !HIP_INCLUDE_HIP_AMD_DETAIL_HIP_TEXTURE_TYPES_H */
