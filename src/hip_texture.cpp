
#include <map>

#include <string.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

#include "hip_texture.h"

static std::map<hipTextureObject_t, hipTexture*> textureHash;

void saveTextureInfo(const hipTexture* pTexture, const hipResourceDesc* pResDesc,
                     const hipTextureDesc* pTexDesc, const hipResourceViewDesc* pResViewDesc) {
    if (pResDesc != nullptr) {
        memcpy((void*)&(pTexture->resDesc), (void*)pResDesc, sizeof(hipResourceDesc));
    }

    if (pTexDesc != nullptr) {
        memcpy((void*)&(pTexture->texDesc), (void*)pTexDesc, sizeof(hipTextureDesc));
    }

    if (pResViewDesc != nullptr) {
        memcpy((void*)&(pTexture->resViewDesc), (void*)pResViewDesc, sizeof(hipResourceViewDesc));
    }
}

void getDrvChannelOrderAndType(const enum hipArray_Format Format, unsigned int NumChannels,
                               hsa_ext_image_channel_order_t* channelOrder,
                               hsa_ext_image_channel_type_t* channelType) {
    switch (Format) {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
            break;
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
            break;
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
            break;
        case HIP_AD_FORMAT_SIGNED_INT8:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
            break;
        case HIP_AD_FORMAT_SIGNED_INT16:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
            break;
        case HIP_AD_FORMAT_SIGNED_INT32:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
            break;
        case HIP_AD_FORMAT_HALF:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
            break;
        case HIP_AD_FORMAT_FLOAT:
            *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT;
            break;
        default:
            break;
    }

    if (NumChannels == 4) {
        *channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
    } else if (NumChannels == 2) {
        *channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RG;
    } else if (NumChannels == 1) {
        *channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
    }
}
void getChannelOrderAndType(const hipChannelFormatDesc& desc, enum hipTextureReadMode readMode,
                            hsa_ext_image_channel_order_t* channelOrder,
                            hsa_ext_image_channel_type_t* channelType) {
    if (desc.x != 0 && desc.y != 0 && desc.z != 0 && desc.w != 0) {
        *channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
    } else if (desc.x != 0 && desc.y != 0 && desc.z != 0 && desc.w == 0) {
        *channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RGB;
    } else if (desc.x != 0 && desc.y != 0 && desc.z == 0 && desc.w == 0) {
        *channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RG;
    } else if (desc.x != 0 && desc.y == 0 && desc.z == 0 && desc.w == 0) {
        *channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
    } else {
    }

    switch (desc.f) {
        case hipChannelFormatKindUnsigned:
            switch (desc.x) {
                case 32:
                    *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
                    break;
                case 16:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16
                                       : HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
                    break;
                case 8:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8
                                       : HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
                    break;
                default:
                    *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
            }
            break;
        case hipChannelFormatKindSigned:
            switch (desc.x) {
                case 32:
                    *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
                    break;
                case 16:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16
                                       : HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
                    break;
                case 8:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8
                                       : HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
                    break;
                default:
                    *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
            }
            break;
        case hipChannelFormatKindFloat:
            switch (desc.x) {
                case 32:
                    *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT;
                    break;
                case 16:
                    *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
                    break;
                case 8:
                    break;
                default:
                    *channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT;
            }
            break;
        case hipChannelFormatKindNone:
        default:
            break;
    }
}

void fillSamplerDescriptor(hsa_ext_sampler_descriptor_t& samplerDescriptor,
                           enum hipTextureAddressMode addressMode,
                           enum hipTextureFilterMode filterMode, int normalizedCoords) {
    if (normalizedCoords) {
        samplerDescriptor.coordinate_mode = HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED;
    } else {
        samplerDescriptor.coordinate_mode = HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED;
    }

    switch (filterMode) {
        case hipFilterModePoint:
            samplerDescriptor.filter_mode = HSA_EXT_SAMPLER_FILTER_MODE_NEAREST;
            break;
        case hipFilterModeLinear:
            samplerDescriptor.filter_mode = HSA_EXT_SAMPLER_FILTER_MODE_LINEAR;
            break;
    }

    switch (addressMode) {
        case hipAddressModeWrap:
            samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT;
            break;
        case hipAddressModeClamp:
            samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
            break;
        case hipAddressModeMirror:
            samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
            break;
        case hipAddressModeBorder:
            samplerDescriptor.address_mode = HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER;
            break;
    }
}

bool getHipTextureObject(hipTextureObject_t* pTexObject, hsa_ext_image_t& image,
                         hsa_ext_sampler_t sampler) {
    unsigned int* texSRD;
    hipMalloc((void**)&texSRD, HIP_TEXTURE_OBJECT_SIZE_DWORD * 4);
    hipMemcpy(texSRD, (void*)image.handle, HIP_IMAGE_OBJECT_SIZE_DWORD * 4,
              hipMemcpyDeviceToDevice);
    hipMemcpy(texSRD + HIP_SAMPLER_OBJECT_OFFSET_DWORD, (void*)sampler.handle,
              HIP_SAMPLER_OBJECT_SIZE_DWORD * 4, hipMemcpyDeviceToDevice);
    *pTexObject = (hipTextureObject_t)texSRD;

#ifdef DEBUG
    unsigned int* srd = (unsigned int*)malloc(HIP_TEXTURE_OBJECT_SIZE_DWORD * 4);
    hipMemcpy(srd, texSRD, HIP_TEXTURE_OBJECT_SIZE_DWORD * 4, hipMemcpyDeviceToHost);
    printf("New SRD: \n");
    for (int i = 0; i < HIP_TEXTURE_OBJECT_SIZE_DWORD; i++) {
        printf("SRD[%d]: %x\n", i, srd[i]);
    }
    printf("\n");
#endif
    return true;
}

// Texture Object APIs
hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const hipResourceViewDesc* pResViewDesc) {
    HIP_INIT_API(hipCreateTextureObject, pTexObject, pResDesc, pTexDesc, pResViewDesc);
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hc::accelerator acc = ctx->getDevice()->_acc;
        auto device = ctx->getWriteableDevice();

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        hipTexture* pTexture = (hipTexture*)malloc(sizeof(hipTexture));
        if (pTexture != nullptr) {
            memset(pTexture, 0, sizeof(hipTexture));
            saveTextureInfo(pTexture, pResDesc, pTexDesc, pResViewDesc);
        }

        hsa_ext_image_descriptor_t imageDescriptor;
        hsa_ext_image_channel_order_t channelOrder;
        hsa_ext_image_channel_type_t channelType;
        void* devPtr = nullptr;

        switch (pResDesc->resType) {
            case hipResourceTypeArray:
                devPtr = pResDesc->res.array.array->data;
                imageDescriptor.width = pResDesc->res.array.array->width;
                imageDescriptor.height = pResDesc->res.array.array->height;
                switch (pResDesc->res.array.array->type) {
                    case hipArrayLayered:
                        imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
                        imageDescriptor.depth = 0;
                        imageDescriptor.array_size = pResDesc->res.array.array->depth;
                        break;
                    case hipArrayCubemap:
                        imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
                        imageDescriptor.depth = pResDesc->res.array.array->depth;
                        imageDescriptor.array_size = 0;
                        break;
                    case hipArraySurfaceLoadStore:
                    case hipArrayTextureGather:
                    case hipArrayDefault:
                    default:
                        imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
                        imageDescriptor.depth = 0;
                        imageDescriptor.array_size = 0;
                        break;
                }
                getChannelOrderAndType(pResDesc->res.array.array->desc, pTexDesc->readMode,
                                       &channelOrder, &channelType);
                break;
            case hipResourceTypeMipmappedArray:
                devPtr = pResDesc->res.mipmap.mipmap->data;
                imageDescriptor.width = pResDesc->res.mipmap.mipmap->width;
                imageDescriptor.height = pResDesc->res.mipmap.mipmap->height;
                imageDescriptor.depth = pResDesc->res.mipmap.mipmap->depth;
                imageDescriptor.array_size = 0;
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
                getChannelOrderAndType(pResDesc->res.mipmap.mipmap->desc, pTexDesc->readMode,
                                       &channelOrder, &channelType);
                break;
            case hipResourceTypeLinear:
                devPtr = pResDesc->res.linear.devPtr;
                imageDescriptor.width = pResDesc->res.linear.sizeInBytes/((pResDesc->res.linear.desc.x + pResDesc->res.linear.desc.y + pResDesc->res.linear.desc.z + pResDesc->res.linear.desc.w)/8);
                imageDescriptor.height = 1;
                imageDescriptor.depth = 0;
                imageDescriptor.array_size = 0;
                imageDescriptor.geometry =
                    HSA_EXT_IMAGE_GEOMETRY_1D;  // ? HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR
                getChannelOrderAndType(pResDesc->res.linear.desc, pTexDesc->readMode, &channelOrder,
                                       &channelType);
                break;
            case hipResourceTypePitch2D:
                devPtr = pResDesc->res.pitch2D.devPtr;
                imageDescriptor.width = pResDesc->res.pitch2D.width;
                imageDescriptor.height = pResDesc->res.pitch2D.height;
                imageDescriptor.depth = 0;
                imageDescriptor.array_size = 0;
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
                getChannelOrderAndType(pResDesc->res.pitch2D.desc, pTexDesc->readMode,
                                       &channelOrder, &channelType);
                break;
            default:
                break;
        }

        imageDescriptor.format.channel_order = channelOrder;
        imageDescriptor.format.channel_type = channelType;

        hsa_ext_sampler_descriptor_t samplerDescriptor;
        fillSamplerDescriptor(samplerDescriptor, pTexDesc->addressMode[0], pTexDesc->filterMode,
                              pTexDesc->normalizedCoords);

        hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
        if (HSA_STATUS_SUCCESS != hsa_ext_image_create_with_layout(
                                      *agent, &imageDescriptor, devPtr, permission,
                                      HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &(pTexture->image)) ||
            HSA_STATUS_SUCCESS !=
                hsa_ext_sampler_create(*agent, &samplerDescriptor, &(pTexture->sampler))) {
            return ihipLogStatus(hipErrorRuntimeOther);
        }

        getHipTextureObject(pTexObject, pTexture->image, pTexture->sampler);

        textureHash[*pTexObject] = pTexture;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
    HIP_INIT_API(hipDestroyTextureObject, textureObject);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hc::accelerator acc = ctx->getDevice()->_acc;
        auto device = ctx->getWriteableDevice();

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        hipTexture* pTexture = textureHash[textureObject];
        if (pTexture != nullptr) {
            hsa_ext_image_destroy(*agent, pTexture->image);
            hsa_ext_sampler_destroy(*agent, pTexture->sampler);
            free(pTexture);
            textureHash.erase(textureObject);
        }
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
    HIP_INIT_API(hipGetTextureObjectResourceDesc, pResDesc, textureObject);
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hipTexture* pTexture = textureHash[textureObject];
        if (pTexture != nullptr && pResDesc != nullptr) {
            memcpy((void*)pResDesc, (void*)&(pTexture->resDesc), sizeof(hipResourceDesc));
        }
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject) {
    HIP_INIT_API(hipGetTextureObjectResourceViewDesc, pResViewDesc, textureObject);
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hipTexture* pTexture = textureHash[textureObject];
        if (pTexture != nullptr && pResViewDesc != nullptr) {
            memcpy((void*)pResViewDesc, (void*)&(pTexture->resViewDesc),
                   sizeof(hipResourceViewDesc));
        }
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject) {
    HIP_INIT_API(hipGetTextureObjectTextureDesc, pTexDesc, textureObject);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hipTexture* pTexture = textureHash[textureObject];
        if (pTexture != nullptr && pTexDesc != nullptr) {
            memcpy((void*)pTexDesc, (void*)&(pTexture->texDesc), sizeof(hipTextureDesc));
        }
    }
    return ihipLogStatus(hip_status);
}

// Texture Reference APIs
hipError_t ihipBindTextureImpl(int dim, enum hipTextureReadMode readMode, size_t* offset,
                               const void* devPtr, const struct hipChannelFormatDesc* desc,
                               size_t size, textureReference* tex) {
    hipError_t hip_status = hipSuccess;
    enum hipTextureAddressMode addressMode = tex->addressMode[0];
    enum hipTextureFilterMode filterMode = tex->filterMode;
    int normalizedCoords = tex->normalized;
    hipTextureObject_t& textureObject = tex->textureObject;
    if(offset != nullptr)
        *offset = 0;
    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hc::accelerator acc = ctx->getDevice()->_acc;
        auto device = ctx->getWriteableDevice();

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        hipTexture* pTexture = (hipTexture*)malloc(sizeof(hipTexture));
        if (pTexture != nullptr) {
            memset(pTexture, 0, sizeof(hipTexture));
        }

        hsa_ext_image_descriptor_t imageDescriptor;

        assert(dim == hipTextureType1D);

        imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_1D;
        imageDescriptor.width = size;
        imageDescriptor.height = 1;
        imageDescriptor.depth = 1;
        imageDescriptor.array_size = 0;

        hsa_ext_image_channel_order_t channelOrder;
        hsa_ext_image_channel_type_t channelType;
        if (NULL == desc) {
            getDrvChannelOrderAndType(tex->format, tex->numChannels, &channelOrder, &channelType);
        } else {
            getChannelOrderAndType(*desc, readMode, &channelOrder, &channelType);
        }
        imageDescriptor.format.channel_order = channelOrder;
        imageDescriptor.format.channel_type = channelType;

        hsa_ext_sampler_descriptor_t samplerDescriptor;
        fillSamplerDescriptor(samplerDescriptor, addressMode, filterMode, normalizedCoords);

        hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;

        if (HSA_STATUS_SUCCESS != hsa_ext_image_create_with_layout(
                                      *agent, &imageDescriptor, devPtr, permission,
                                      HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &(pTexture->image)) ||
            HSA_STATUS_SUCCESS !=
                hsa_ext_sampler_create(*agent, &samplerDescriptor, &(pTexture->sampler))) {
            return hipErrorRuntimeOther;
        }
        getHipTextureObject(&textureObject, pTexture->image, pTexture->sampler);
        textureHash[textureObject] = pTexture;
    }

    return hip_status;
}

hipError_t hipBindTexture(size_t* offset, textureReference* tex, const void* devPtr,
                          const hipChannelFormatDesc* desc, size_t size) {
    HIP_INIT_API(hipBindTexture, offset, tex, devPtr, desc, size);
    hipError_t hip_status = hipSuccess;
    // TODO: hipReadModeElementType is default.
    hip_status = ihipBindTextureImpl(hipTextureType1D, hipReadModeElementType, offset, devPtr, desc,
                                     size, tex);
    return ihipLogStatus(hip_status);
}

hipError_t ihipBindTexture2DImpl(int dim, enum hipTextureReadMode readMode, size_t* offset,
                                 const void* devPtr, const struct hipChannelFormatDesc* desc,
                                 size_t width, size_t height, textureReference* tex) {
    hipError_t hip_status = hipSuccess;
    enum hipTextureAddressMode addressMode = tex->addressMode[0];
    enum hipTextureFilterMode filterMode = tex->filterMode;
    int normalizedCoords = tex->normalized;
    hipTextureObject_t& textureObject = tex->textureObject;
    if(offset != nullptr)
        *offset = 0;
    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hc::accelerator acc = ctx->getDevice()->_acc;
        auto device = ctx->getWriteableDevice();

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        hipTexture* pTexture = (hipTexture*)malloc(sizeof(hipTexture));
        if (pTexture != nullptr) {
            memset(pTexture, 0, sizeof(hipTexture));
        }

        hsa_ext_image_descriptor_t imageDescriptor;

        assert(dim == hipTextureType2D);

        imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
        imageDescriptor.width = width;
        imageDescriptor.height = height;
        imageDescriptor.depth = 1;
        imageDescriptor.array_size = 0;

        hsa_ext_image_channel_order_t channelOrder;
        hsa_ext_image_channel_type_t channelType;

        if (NULL == desc) {
            getDrvChannelOrderAndType(tex->format, tex->numChannels, &channelOrder, &channelType);
        } else {
            getChannelOrderAndType(*desc, readMode, &channelOrder, &channelType);
        }
        imageDescriptor.format.channel_order = channelOrder;
        imageDescriptor.format.channel_type = channelType;

        hsa_ext_sampler_descriptor_t samplerDescriptor;
        fillSamplerDescriptor(samplerDescriptor, addressMode, filterMode, normalizedCoords);

        hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;

        if (HSA_STATUS_SUCCESS != hsa_ext_image_create_with_layout(
                                      *agent, &imageDescriptor, devPtr, permission,
                                      HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &(pTexture->image)) ||
            HSA_STATUS_SUCCESS !=
                hsa_ext_sampler_create(*agent, &samplerDescriptor, &(pTexture->sampler))) {
            return hipErrorRuntimeOther;
        }
        getHipTextureObject(&textureObject, pTexture->image, pTexture->sampler);
        textureHash[textureObject] = pTexture;
    }

    return hip_status;
}

hipError_t hipBindTexture2D(size_t* offset, textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch) {
    HIP_INIT_API(hipBindTexture2D, offset, tex, devPtr, desc, width, height, pitch);
    hipError_t hip_status = hipSuccess;
    hip_status = ihipBindTexture2DImpl(hipTextureType2D, hipReadModeElementType, offset, devPtr,
                                       desc, width, height, tex);
    return ihipLogStatus(hip_status);
}

hipError_t ihipBindTextureToArrayImpl(int dim, enum hipTextureReadMode readMode,
                                      hipArray_const_t array,
                                      const struct hipChannelFormatDesc& desc,
                                      textureReference* tex) {
    hipError_t hip_status = hipSuccess;
    enum hipTextureAddressMode addressMode = tex->addressMode[0];
    enum hipTextureFilterMode filterMode = tex->filterMode;
    int normalizedCoords = tex->normalized;
    hipTextureObject_t& textureObject = tex->textureObject;
    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hc::accelerator acc = ctx->getDevice()->_acc;
        auto device = ctx->getWriteableDevice();

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        hipTexture* pTexture = (hipTexture*)malloc(sizeof(hipTexture));
        if (pTexture != nullptr) {
            memset(pTexture, 0, sizeof(hipTexture));
        }

        hsa_ext_image_descriptor_t imageDescriptor;

        imageDescriptor.width = array->width;
        imageDescriptor.height = array->height;
        imageDescriptor.depth = array->depth;
        imageDescriptor.array_size = 0;

        switch (dim) {
            case hipTextureType1D:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_1D;
                imageDescriptor.height = 1;
                imageDescriptor.depth = 1;
                break;
            case hipTextureType2D:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
                imageDescriptor.depth = 1;
                break;
            case hipTextureType3D:
            case hipTextureTypeCubemap:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
                break;
            case hipTextureType1DLayered:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_1DA;
                imageDescriptor.height = 1;
                imageDescriptor.array_size = array->height;
                break;
            case hipTextureType2DLayered:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
                imageDescriptor.depth = 1;
                imageDescriptor.array_size = array->depth;
                break;
            case hipTextureTypeCubemapLayered:
            default:
                break;
        }

        hsa_ext_image_channel_order_t channelOrder;
        hsa_ext_image_channel_type_t channelType;
        if (array->isDrv) {
            getDrvChannelOrderAndType(array->drvDesc.format, array->drvDesc.numChannels,
                                      &channelOrder, &channelType);
        } else {
            getChannelOrderAndType(desc, readMode, &channelOrder, &channelType);
        }
        imageDescriptor.format.channel_order = channelOrder;
        imageDescriptor.format.channel_type = channelType;

        hsa_ext_sampler_descriptor_t samplerDescriptor;
        fillSamplerDescriptor(samplerDescriptor, addressMode, filterMode, normalizedCoords);

        hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;

        if (HSA_STATUS_SUCCESS != hsa_ext_image_create_with_layout(
                                      *agent, &imageDescriptor, array->data, permission,
                                      HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &(pTexture->image)) ||
            HSA_STATUS_SUCCESS !=
                hsa_ext_sampler_create(*agent, &samplerDescriptor, &(pTexture->sampler))) {
            return hipErrorRuntimeOther;
        }
        getHipTextureObject(&textureObject, pTexture->image, pTexture->sampler);
        textureHash[textureObject] = pTexture;
    }

    return hip_status;
}

hipError_t hipBindTextureToArray(textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
    HIP_INIT_API(hipBindTextureToArray, tex, array, desc);
    hipError_t hip_status = hipSuccess;
    // TODO: hipReadModeElementType is default.
    hip_status =
        ihipBindTextureToArrayImpl(array->textureType, hipReadModeElementType, array, *desc, tex);
    return ihipLogStatus(hip_status);
}

hipError_t hipBindTextureToMipmappedArray(textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
    HIP_INIT_API(hipBindTextureToMipmappedArray, tex, mipmappedArray, desc);
    hipError_t hip_status = hipSuccess;
    return ihipLogStatus(hip_status);
}

hipError_t ihipUnbindTextureImpl(const hipTextureObject_t& textureObject) {
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        hc::accelerator acc = ctx->getDevice()->_acc;
        auto device = ctx->getWriteableDevice();

        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        hipTexture* pTexture = textureHash[textureObject];
        if (pTexture != nullptr) {
            hsa_ext_image_destroy(*agent, pTexture->image);
            hsa_ext_sampler_destroy(*agent, pTexture->sampler);
            free(pTexture);
            textureHash.erase(textureObject);
        }
    }

    return hip_status;
}

hipError_t hipUnbindTexture(const textureReference* tex) {
    HIP_INIT_API(hipUnbindTexture, tex);
    hipError_t hip_status = hipSuccess;
    hip_status = ihipUnbindTextureImpl(tex->textureObject);
    return ihipLogStatus(hip_status);
}

hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array) {
    HIP_INIT_API(hipGetChannelDesc, desc, array);
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        *desc = array->desc;
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* tex) {
    HIP_INIT_API(hipGetTextureAlignmentOffset, offset, tex);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        if(offset != nullptr)
            *offset = 0;
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipGetTextureReference(const textureReference** tex, const void* symbol) {
    HIP_INIT_API(hipGetTextureReference, tex, symbol);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipTexRefSetFormat(textureReference* tex, hipArray_Format fmt, int NumPackedComponents) {
    HIP_INIT_API(hipTexRefSetFormat, tex, fmt, NumPackedComponents);
    hipError_t hip_status = hipSuccess;
    tex->format = fmt;
    tex->numChannels = NumPackedComponents;
    return ihipLogStatus(hip_status);
}

hipError_t hipTexRefSetFlags(textureReference* tex, unsigned int flags) {
    HIP_INIT_API(hipTexRefSetFlags, tex, flags);
    hipError_t hip_status = hipSuccess;
    tex->normalized = flags;
    return ihipLogStatus(hip_status);
}

hipError_t hipTexRefSetFilterMode(textureReference* tex, hipTextureFilterMode fm) {
    HIP_INIT_API(hipTexRefSetFilterMode, tex, fm);
    hipError_t hip_status = hipSuccess;
    tex->filterMode = fm;
    return ihipLogStatus(hip_status);
}

hipError_t hipTexRefSetAddressMode(textureReference* tex, int dim, hipTextureAddressMode am) {
    HIP_INIT_API(hipTexRefSetAddressMode, tex, dim, am);
    hipError_t hip_status = hipSuccess;
    tex->addressMode[dim] = am;
    return ihipLogStatus(hip_status);
}

hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags) {
    HIP_INIT_API(hipTexRefSetArray, tex, array, flags);
    hipError_t hip_status = hipSuccess;

    hip_status = ihipBindTextureToArrayImpl(array->textureType, hipReadModeElementType, array,
                                            array->desc, tex);
    return ihipLogStatus(hip_status);
}


hipError_t hipTexRefSetAddress(size_t* offset, textureReference* tex, hipDeviceptr_t devPtr,
                               size_t size) {
    HIP_INIT_API(hipTexRefSetAddress, offset, tex, devPtr, size);
    hipError_t hip_status = hipSuccess;
    // TODO: hipReadModeElementType is default.
    hip_status = ihipBindTextureImpl(hipTextureType1D, hipReadModeElementType, offset, devPtr, NULL,
                                     size, tex);
    return ihipLogStatus(hip_status);
}

hipError_t hipTexRefSetAddress2D(textureReference* tex, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t devPtr, size_t pitch) {
    HIP_INIT_API(hipTexRefSetAddress2D, tex, desc, devPtr, pitch);
    size_t offset;
    hipError_t hip_status = hipSuccess;
    // TODO: hipReadModeElementType is default.
    hip_status = ihipBindTexture2DImpl(hipTextureType2D, hipReadModeElementType, &offset, devPtr,
                                       NULL, desc->width, desc->height, tex);
    return ihipLogStatus(hip_status);
}
