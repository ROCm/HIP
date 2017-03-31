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

//#pragma once

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_TEXTURE_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_TEXTURE_H

/**
 *  @file  hcc_detail/hip_texture.h
 *  @brief HIP C++ Texture API for hcc compiler
 */

#include <limits.h>
#include <hip/hcc_detail/driver_types.h>
#include <hip/hcc_detail/channel_descriptor.h>
#include <hip/hcc_detail/texture_types.h>
//#include <hip/hcc_detail/hip_runtime.h>

//----
//Texture - TODO - likely need to move this to a separate file only included with kernel compilation.
#define hipTextureType1D 1

#if __cplusplus
template <class T, int texType=hipTextureType1D, hipTextureReadMode readMode=hipReadModeElementType>
struct texture : public textureReference {

    const T * _dataPtr;  // pointer to underlying data.

    //texture() : filterMode(hipFilterModePoint), normalized(false), _dataPtr(NULL) {};
    unsigned int width;
    unsigned int height;

};
#endif


#define tex1Dfetch(_tex, _addr) (_tex._dataPtr[_addr])

#define tex2D(_tex, _dx, _dy) \
  _tex._dataPtr[(unsigned int)_dx + (unsigned int)_dy*(_tex.width)]

/**
 *  @addtogroup API HIP API
 *  @{
 *
 *  Defines the HIP API.  See the individual sections for more information.
 */

// These are C++ APIs - maybe belong in separate file.
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Texture Texture Reference Management
 *  @{
 *
 *
 *  @warning The HIP texture API implements a small subset of full texture API.  Known limitations include:
 *  - Only point sampling is supported.
 *  - Only C++ APIs are provided.
 *  - Many APIs and modes are not implemented.
 *
 *  The HIP texture support is intended to allow use of texture cache on hardware where this is beneficial.
 *
 *  The following CUDA APIs are not currently supported:
 *  - cudaBindTexture2D
 *  - cudaBindTextureToArray
 *  - cudaBindTextureToMipmappedArray
 *  - cudaGetChannelDesc
 *  - cudaGetTextureReference
 *
 */

// C API:
#if 0
hipChannelFormatDesc  hipBindTexture(size_t *offset, struct textureReference *tex, const void *devPtr, const struct hipChannelFormatDesc *desc, size_t size=UINT_MAX)
{
    tex->_dataPtr = devPtr;
}
#endif


// End doxygen API:
/**
 *   @}
 */

#endif
