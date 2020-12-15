/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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


#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_TEXTURE_TYPES_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_TEXTURE_TYPES_H

#include <hip/amd_detail/driver_types.h>

#define hipTextureType1D 0x01
#define hipTextureType2D 0x02
#define hipTextureType3D 0x03
#define hipTextureTypeCubemap 0x0C
#define hipTextureType1DLayered 0xF1
#define hipTextureType2DLayered 0xF2
#define hipTextureTypeCubemapLayered 0xFC

/**
 * Should be same as HSA_IMAGE_OBJECT_SIZE_DWORD/HSA_SAMPLER_OBJECT_SIZE_DWORD
 */
#define HIP_IMAGE_OBJECT_SIZE_DWORD 12
#define HIP_SAMPLER_OBJECT_SIZE_DWORD 8
#define HIP_SAMPLER_OBJECT_OFFSET_DWORD HIP_IMAGE_OBJECT_SIZE_DWORD
#define HIP_TEXTURE_OBJECT_SIZE_DWORD (HIP_IMAGE_OBJECT_SIZE_DWORD + HIP_SAMPLER_OBJECT_SIZE_DWORD)

/**
 * An opaque value that represents a hip texture object
 */
struct __hip_texture;
typedef struct __hip_texture* hipTextureObject_t;

/**
 * hip texture address modes
 */
enum hipTextureAddressMode {
    hipAddressModeWrap = 0,
    hipAddressModeClamp = 1,
    hipAddressModeMirror = 2,
    hipAddressModeBorder = 3
};

/**
 * hip texture filter modes
 */
enum hipTextureFilterMode { hipFilterModePoint = 0, hipFilterModeLinear = 1 };

/**
 * hip texture read modes
 */
enum hipTextureReadMode { hipReadModeElementType = 0, hipReadModeNormalizedFloat = 1 };

/**
 * hip texture reference
 */
typedef struct textureReference {
    int normalized;
    enum hipTextureReadMode readMode;// used only for driver API's
    enum hipTextureFilterMode filterMode;
    enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
    struct hipChannelFormatDesc channelDesc;
    int sRGB;                    // Perform sRGB->linear conversion during texture read
    unsigned int maxAnisotropy;  // Limit to the anisotropy ratio
    enum hipTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;

    hipTextureObject_t textureObject;
    int numChannels;
    enum hipArray_Format format;
}textureReference;

/**
 * hip texture descriptor
 */
typedef struct hipTextureDesc {
    enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
    enum hipTextureFilterMode filterMode;
    enum hipTextureReadMode readMode;
    int sRGB;  // Perform sRGB->linear conversion during texture read
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    enum hipTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
}hipTextureDesc;

#endif
