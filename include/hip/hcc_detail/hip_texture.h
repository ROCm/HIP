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

//#pragma once

#ifndef HIP_TEXTURE_H
#define HIP_TEXTURE_H

/**
 *  @file  hcc_detail/hip_texture.h
 *  @brief HIP C++ Texture API for hcc compiler
 */

#include <limits.h>

#include <hip/hcc_detail/hip_runtime.h>

//----
//Texture - TODO - likely need to move this to a separate file only included with kernel compilation.
#define hipTextureType1D 1

typedef enum {
  hipChannelFormatKindSigned = 0,
  hipChannelFormatKindUnsigned,
  hipChannelFormatKindFloat,
  hipChannelFormatKindNone

} hipChannelFormatKind;

typedef struct hipChannelFormatDesc {
  int x;
  int y;
  int z;
  int w;
  hipChannelFormatKind f;
} hipChannelFormatDesc;

typedef enum hipTextureReadMode
{
  hipReadModeElementType,  ///< Read texture as specified element type
//! @warning cudaReadModeNormalizedFloat is not supported.
} hipTextureReadMode;

typedef enum hipTextureFilterMode
{
    hipFilterModePoint,  ///< Point filter mode.
//! @warning cudaFilterModeLinear is not supported.
} hipTextureFilterMode;

struct textureReference {
    hipTextureFilterMode filterMode;
    bool                 normalized;
    hipChannelFormatDesc channelDesc;
};
#if __cplusplus
template <class T, int texType=hipTextureType1D, enum hipTextureReadMode=hipReadModeElementType>
struct texture : public textureReference {

    const T * _dataPtr;  // pointer to underlying data.

    //texture() : filterMode(hipFilterModePoint), normalized(false), _dataPtr(NULL) {};
    unsigned int width;
    unsigned int height;

};
#endif

typedef struct hipArray {
  unsigned int width;
  unsigned int height;
  hipChannelFormatKind f;
  void* data; //FIXME: generalize this
} hipArray;


#define tex1Dfetch(_tex, _addr) (_tex._dataPtr[_addr])

#define tex2D(_tex, _dx, _dy) \
  _tex._dataPtr[(unsigned int)_dx + (unsigned int)_dy*(_tex.width)]

/**
 *  @brief Allocate an array on the device.
 *
 *  @param[out]  array  Pointer to allocated array in device memory
 *  @param[in]   desc   Requested channel format
 *  @param[in]   width  Requested array allocation width
 *  @param[in]   height Requested array allocation height
 *  @param[in]   flags  Requested properties of allocated array
 *  @return      #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
 */
hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                          size_t width, size_t height = 0, unsigned int flags = 0);

/**
 *  @brief Frees an array on the device. 
 *
 *  @param[in]  array  Pointer to array to free
 *  @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
 */
hipError_t hipFreeArray(hipArray* array);

/**
 *  @brief Copies data between host and device. 
 *
 *  @param[in]   dst    Destination memory address
 *  @param[in]   dpitch Pitch of destination memory
 *  @param[in]   src    Source memory address
 *  @param[in]   spitch Pitch of source memory 
 *  @param[in]   width  Width of matrix transfer (columns in bytes)
 *  @param[in]   height Height of matrix transfer (rows)
 *  @param[in]   kind   Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue, #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol, hipMemcpyAsync
 */
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst    Destination memory address
 *  @param[in]   dpitch Pitch of destination memory
 *  @param[in]   src    Source memory address
 *  @param[in]   spitch Pitch of source memory
 *  @param[in]   width  Width of matrix transfer (columns in bytes)
 *  @param[in]   height Height of matrix transfer (rows)
 *  @param[in]   kind   Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue, #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol, hipMemcpyAsync
 */
hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst    Destination memory address
 *  @param[in]   dpitch Pitch of destination memory
 *  @param[in]   src    Source memory address
 *  @param[in]   spitch Pitch of source memory
 *  @param[in]   width  Width of matrix transfer (columns in bytes)
 *  @param[in]   height Height of matrix transfer (rows)
 *  @param[in]   kind   Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue, #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol, hipMemcpyAsync
 */
hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset,
                            const void* src, size_t count, hipMemcpyKind kind);


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

/**
 *  @brief Returns a channel descriptor using the specified format.
 *
 *  @param[in]   x    X component
 *  @param[in]   y    Y component
 *  @param[in]   z    Z component
 *  @param[in]   w    W component
 *  @param[in]   f    Channel format
 *  @return      Channel descriptor with format f 
 *
 */
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);

// descriptors
template <typename T> inline hipChannelFormatDesc hipCreateChannelDesc() {
  return hipCreateChannelDesc(0, 0, 0, 0, hipChannelFormatKindNone);
}
template <> inline hipChannelFormatDesc hipCreateChannelDesc<int>() {
  int e = (int)sizeof(int) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}
template <> inline hipChannelFormatDesc hipCreateChannelDesc<unsigned int>() {
  int e = (int)sizeof(unsigned int) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}
template <> inline hipChannelFormatDesc hipCreateChannelDesc<long>() {
  int e = (int)sizeof(long) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}
template <> inline hipChannelFormatDesc hipCreateChannelDesc<unsigned long>() {
  int e = (int)sizeof(unsigned long) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}
template <> inline hipChannelFormatDesc hipCreateChannelDesc<float>() {
  int e = (int)sizeof(float) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

/*
 * @brief hipBindTexture Binds size bytes of the memory area pointed to by @p devPtr to the texture reference tex.
 *
 * @p desc describes how the memory is interpreted when fetching values from the texture. The @p offset parameter is an optional byte offset as with the low-level
 * hipBindTexture() function. Any memory previously bound to tex is unbound.
 *
 *  @param[in]  offset - Offset in bytes
 *  @param[out]  tex - texture to bind
 *  @param[in]  devPtr - Memory area on device
 *  @param[in]  desc - Channel format
 *  @param[in]  size - Size of the memory area pointed to by devPtr
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 **/
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t  hipBindTexture(size_t *offset,
                                     struct texture<T, dim, readMode> &tex,
                                     const void *devPtr,
                                     const struct hipChannelFormatDesc *desc,
                                     size_t size=UINT_MAX)
{
    tex._dataPtr = static_cast<const T*>(devPtr);

    return hipSuccess;
}

/*
 * @brief hipBindTexture Binds size bytes of the memory area pointed to by @p devPtr to the texture reference tex.
 *
 * @p desc describes how the memory is interpreted when fetching values from the texture. The @p offset parameter is an optional byte offset as with the low-level
 * hipBindTexture() function. Any memory previously bound to tex is unbound.
 *
 *  @param[in]  offset - Offset in bytes
 *  @param[in]  tex - texture to bind
 *  @param[in]  devPtr - Memory area on device
 *  @param[in]  size - Size of the memory area pointed to by devPtr
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 **/
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t  hipBindTexture(size_t *offset,
                                     struct texture<T, dim, readMode> &tex,
                                     const void *devPtr,
                                     size_t size=UINT_MAX)
{
    return  hipBindTexture(offset, tex, devPtr, &tex.channelDesc, size);
}

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTextureToArray(struct texture<T, dim, readMode> &tex, hipArray* array) {
  tex.width = array->width;
  tex.height = array->height;
  tex._dataPtr = static_cast<const T*>(array->data);
  return hipSuccess;
}

/*
 * @brief Unbinds the textuer bound to @p tex
 *
 *  @param[in]  tex - texture to unbind
 *
 *  @return #hipSuccess
 **/
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t  hipUnbindTexture(struct texture<T, dim, readMode> &tex)
{
    tex._dataPtr = NULL;

    return hipSuccess;
}



// doxygen end Texture
/**
 * @}
 */


// End doxygen API:
/**
 *   @}
 */

#endif

