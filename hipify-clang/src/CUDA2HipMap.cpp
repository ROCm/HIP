#include "CUDA2HipMap.h"

/// Maps the names of CUDA types to the corresponding hip types.
const std::map<llvm::StringRef, hipCounter> CUDA_TYPE_NAME_MAP{
    // Error codes and return types
    {"CUresult",    {"hipError_t", CONV_TYPE, API_DRIVER}},
//      {"cudaError_enum", {"hipError_t", CONV_TYPE, API_DRIVER}},
    {"cudaError_t", {"hipError_t", CONV_TYPE, API_RUNTIME}},
    {"cudaError",   {"hipError_t", CONV_TYPE, API_RUNTIME}},

    ///////////////////////////// CUDA DRIVER API /////////////////////////////
    {"CUDA_ARRAY3D_DESCRIPTOR",           {"HIP_ARRAY3D_DESCRIPTOR",           CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUDA_ARRAY_DESCRIPTOR",             {"HIP_ARRAY_DESCRIPTOR",             CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUDA_MEMCPY2D",                     {"HIP_MEMCPY2D",                     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUDA_MEMCPY3D",                     {"HIP_MEMCPY3D",                     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUDA_MEMCPY3D_PEER",                {"HIP_MEMCPY3D_PEER",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUDA_POINTER_ATTRIBUTE_P2P_TOKENS", {"HIP_POINTER_ATTRIBUTE_P2P_TOKENS", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUDA_RESOURCE_DESC",                {"HIP_RESOURCE_DESC",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUDA_RESOURCE_VIEW_DESC",           {"HIP_RESOURCE_VIEW_DESC",           CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUipcEventHandle", {"hipIpcEventHandle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUipcMemHandle",   {"hipIpcMemHandle",   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUaddress_mode",        {"hipAddress_mode",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUarray_cubemap_face",  {"hipArray_cubemap_face", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUarray_format",        {"hipArray_format",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUcomputemode",         {"hipComputemode",        CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_RUNTIME ANALOGUE (cudaComputeMode)
    {"CUmem_advise",          {"hipMemAdvise",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_RUNTIME ANALOGUE (cudaComputeMode)
    {"CUmem_range_attribute", {"hipMemRangeAttribute",  CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_RUNTIME ANALOGUE (cudaMemRangeAttribute)
    {"CUctx_flags",           {"hipCctx_flags",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    // NOTE: CUdevice might be changed to typedef int in the future.
    {"CUdevice",                {"hipDevice_t",          CONV_TYPE, API_DRIVER}},
    {"CUdevice_attribute_enum", {"hipDeviceAttribute_t", CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaDeviceAttr)
    {"CUdevice_attribute",      {"hipDeviceAttribute_t", CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaDeviceAttr)
    {"CUdeviceptr",             {"hipDeviceptr_t",       CONV_TYPE, API_DRIVER}},

    // CUDA: "The types::CUarray and struct ::cudaArray * represent the same data type and may be used interchangeably by casting the two types between each other."
    //    typedef struct cudaArray  *cudaArray_t;
    //    typedef struct CUarray_st *CUarray;
    {"CUarray_st", {"hipArray",   CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaArray)
    {"CUarray",    {"hipArray *", CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaArray_t)

    {"CUdevprop_st", {"hipDeviceProp_t", CONV_TYPE, API_DRIVER}},
    {"CUdevprop",    {"hipDeviceProp_t", CONV_TYPE, API_DRIVER}},

    // pointer to CUfunc_st
    {"CUfunction", {"hipFunction_t", CONV_TYPE, API_DRIVER}},

    // TODO: move "typedef struct ihipModuleSymbol_t *hipFunction_t;" from hcc_details to HIP
    //             typedef struct CUfunc_st          *CUfunction;
    //     {"CUfunc_st", {"ihipModuleSymbol_t", CONV_TYPE, API_DRIVER}},

    // typedef struct CUgraphicsResource_st *CUgraphicsResource;
    {"CUgraphicsResource", {"hipGraphicsResource_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    // typedef struct CUmipmappedArray_st *CUmipmappedArray;
    {"CUmipmappedArray",   {"hipMipmappedArray_t",   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUfunction_attribute",      {"hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUfunction_attribute_enum", {"hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUgraphicsMapResourceFlags",      {"hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsMapFlags)
    {"CUgraphicsMapResourceFlags_enum", {"hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsMapFlags)

    {"CUgraphicsRegisterFlags",      {"hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsRegisterFlags)
    {"CUgraphicsRegisterFlags_enum", {"hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsRegisterFlags)

    {"CUoccupancy_flags",      {"hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CUoccupancy_flags_enum", {"hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)

    {"CUfunc_cache_enum", {"hipFuncCache", CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaFuncCache)
    {"CUfunc_cache",      {"hipFuncCache", CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaFuncCache)

    {"CUipcMem_flags",      {"hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CUipcMem_flags_enum", {"hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)

    {"CUjit_cacheMode",      {"hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CUjit_cacheMode_enum", {"hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUjit_fallback",      {"hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CUjit_fallback_enum", {"hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUjit_option",      {"hipJitOption", CONV_JIT, API_DRIVER}},    // API_Runtime ANALOGUE (no)
    {"CUjit_option_enum", {"hipJitOption", CONV_JIT, API_DRIVER}},

    {"CUjit_target",      {"hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CUjit_target_enum", {"hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUjitInputType",      {"hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CUjitInputType_enum", {"hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUlimit",      {"hipLimit_t", CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaLimit)
    {"CUlimit_enum", {"hipLimit_t", CONV_TYPE, API_DRIVER}},    // API_Runtime ANALOGUE (cudaLimit)

    {"CUmemAttach_flags",      {"hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CUmemAttach_flags_enum", {"hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)

    {"CUmemorytype",      {"hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no - cudaMemoryType is not an analogue)
    {"CUmemorytype_enum", {"hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no - cudaMemoryType is not an analogue)

    {"CUresourcetype",      {"hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaResourceType)
    {"CUresourcetype_enum", {"hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaResourceType)

    {"CUresourceViewFormat",      {"hipResourceViewFormat", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaResourceViewFormat)
    {"CUresourceViewFormat_enum", {"hipResourceViewFormat", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaResourceViewFormat)

    {"CUsharedconfig",      {"hipSharedMemConfig", CONV_TYPE, API_DRIVER}},
    {"CUsharedconfig_enum", {"hipSharedMemConfig", CONV_TYPE, API_DRIVER}},

    {"CUcontext",        {"hipCtx_t",              CONV_TYPE, API_DRIVER}},
    // TODO: move "typedef struct ihipCtx_t *hipCtx_t;" from hcc_details to HIP
    //             typedef struct CUctx_st  *CUcontext;
    //     {"CUctx_st", {"ihipCtx_t", CONV_TYPE, API_DRIVER}},
    {"CUmodule",         {"hipModule_t",           CONV_TYPE, API_DRIVER}},
    // TODO: move "typedef struct ihipModule_t *hipModule_t;" from hcc_details to HIP
    //             typedef struct CUmod_st     *CUmodule;
    //     {"CUmod_st", {"ihipModule_t", CONV_TYPE, API_DRIVER}},
    {"CUstream",         {"hipStream_t",           CONV_TYPE, API_DRIVER}},
    // TODO: move "typedef struct ihipStream_t *hipStream_t;" from hcc_details to HIP
    //             typedef struct CUstream_st *CUstream;
    //     {"CUstream_st", {"ihipStream_t", CONV_TYPE, API_DRIVER}},

    // typedef void (*hipStreamCallback_t)      (hipStream_t stream, hipError_t status, void* userData);
    // typedef void (CUDA_CB *CUstreamCallback) (CUstream hStream, CUresult status, void* userData)
    {"CUstreamCallback", {"hipStreamCallback_t",   CONV_TYPE, API_DRIVER}},

    {"CUsurfObject",     {"hipSurfaceObject",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    // typedef struct CUsurfref_st *CUsurfref;
    {"CUsurfref",        {"hipSurfaceReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    //     {"CUsurfref_st", {"ihipSurfaceReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CUtexObject",      {"hipTextureObject",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    // typedef struct CUtexref_st *CUtexref;
    {"CUtexref",         {"hipTextureReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    //     {"CUtexref_st", {"ihipTextureReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    // Stream Flags enum
    {"CUstream_flags",   {"hipStreamFlags",        CONV_TYPE, API_DRIVER}},
    // TODO: ..?
    //     {"CUstream_flags_enum", {"hipStreamFlags", CONV_TYPE, API_DRIVER}},

    {"CUstreamWaitValue_flags", {"hipStreamWaitValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    // TODO: ..?
    //     {"CUstreamWaitValue_flags_enum", {"hipStreamWaitValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUstreamWriteValue_flags", {"hipStreamWriteValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    //     {"CUstreamWriteValue_flags", {"hipStreamWriteValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUstreamBatchMemOpType", {"hipStreamBatchMemOpType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    //     {"CUstreamBatchMemOpType_enum", {"hipStreamBatchMemOpType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    // P2P Attributes
    {"CUdevice_P2PAttribute", {"hipDeviceP2PAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaDeviceP2PAttr)
    //     {"CUdevice_P2PAttribute_enum", {"hipDeviceP2PAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    // pointer to CUevent_st
    {"CUevent",       {"hipEvent_t",    CONV_TYPE,  API_DRIVER}},
    // ToDo:
    //     {"CUevent_st", {"XXXX", CONV_TYPE, API_DRIVER}},
    // Event Flags
    {"CUevent_flags", {"hipEventFlags", CONV_EVENT, API_DRIVER, HIP_UNSUPPORTED}},
    // ToDo:
    //     {"CUevent_flags_enum", {"hipEventFlags", CONV_EVENT, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUfilter_mode", {"hipTextureFilterMode", CONV_TEX, API_DRIVER}},    // API_Runtime ANALOGUE (cudaTextureFilterMode)
    // ToDo:
    //     {"CUfilter_mode", {"CUfilter_mode_enum", CONV_TEX, API_DRIVER}},    // API_Runtime ANALOGUE (cudaTextureFilterMode)

    {"CUGLDeviceList", {"hipGLDeviceList", CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLDeviceList)
    //     {"CUGLDeviceList_enum", {"hipGLDeviceList", CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUGLmap_flags", {"hipGLMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLMapFlags)
    //     {"CUGLmap_flags_enum", {"hipGLMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUd3d9DeviceList", {"hipD3D9DeviceList", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9DeviceList)
    //     {"CUd3d9DeviceList_enum", {"hipD3D9DeviceList", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUd3d9map_flags", {"hipD3D9MapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9MapFlags)
    //     {"CUd3d9map_flags_enum", {"hipD3D9MapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUd3d9register_flags", {"hipD3D9RegisterFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9RegisterFlags)
    //     {"CUd3d9register_flags_enum", {"hipD3D9RegisterFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUd3d10DeviceList", {"hipd3d10DeviceList", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10DeviceList)
    //     {"CUd3d10DeviceList_enum", {"hipD3D10DeviceList", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUd3d10map_flags", {"hipD3D10MapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10MapFlags)
    //     {"CUd3d10map_flags_enum", {"hipD3D10MapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUd3d10register_flags", {"hipD3D10RegisterFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10RegisterFlags)
    //     {"CUd3d10register_flags_enum", {"hipD3D10RegisterFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},

    {"CUd3d11DeviceList", {"hipd3d11DeviceList", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D11DeviceList)
    //     {"CUd3d11DeviceList_enum", {"hipD3D11DeviceList", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},

    // EGL Interoperability
    {"CUeglStreamConnection_st", {"hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEglStreamConnection)
    {"CUeglStreamConnection",    {"hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEglStreamConnection)

    /////////////////////////////// CUDA RT API ///////////////////////////////
    {"libraryPropertyType_t", {"hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"libraryPropertyType",   {"hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

    {"cudaStreamCallback_t", {"hipStreamCallback_t", CONV_TYPE, API_RUNTIME}},

    // Arrays
    {"cudaArray",                  {"hipArray",                  CONV_MEM, API_RUNTIME}},
    // typedef struct cudaArray *cudaArray_t;

    {"cudaArray_t",                {"hipArray_t",                CONV_MEM, API_RUNTIME}},
    // typedef const struct cudaArray *cudaArray_const_t;

    {"cudaArray_const_t",          {"hipArray_const_t",          CONV_MEM, API_RUNTIME}},
    {"cudaMipmappedArray_t",       {"hipMipmappedArray_t",       CONV_MEM, API_RUNTIME}},
    {"cudaMipmappedArray_const_t", {"hipMipmappedArray_const_t", CONV_MEM, API_RUNTIME}},

    {"cudaMemoryAdvise", {"hipMemAdvise", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUmem_advise)
    {"cudaMemRangeAttribute", {"hipMemRangeAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUmem_range_attribute)
    {"cudaMemcpyKind", {"hipMemcpyKind", CONV_MEM, API_RUNTIME}},
    {"cudaMemoryType", {"hipMemoryType", CONV_MEM, API_RUNTIME}},    // API_Driver ANALOGUE (no -  CUmemorytype is not an analogue)

    {"cudaExtent",     {"hipExtent",     CONV_MEM, API_RUNTIME}},
    {"cudaPitchedPtr", {"hipPitchedPtr", CONV_MEM, API_RUNTIME}},
    {"cudaPos",        {"hipPos",        CONV_MEM, API_RUNTIME}},

    {"cudaEvent_t",           {"hipEvent_t",            CONV_TYPE, API_RUNTIME}},
    {"cudaStream_t",          {"hipStream_t",           CONV_TYPE, API_RUNTIME}},
    {"cudaPointerAttributes", {"hipPointerAttribute_t", CONV_TYPE, API_RUNTIME}},

    {"cudaDeviceAttr",      {"hipDeviceAttribute_t",  CONV_TYPE,  API_RUNTIME}},                     // API_DRIVER ANALOGUE (CUdevice_attribute)
    {"cudaDeviceProp",      {"hipDeviceProp_t",       CONV_TYPE,  API_RUNTIME}},
    {"cudaDeviceP2PAttr",   {"hipDeviceP2PAttribute", CONV_TYPE,  API_RUNTIME, HIP_UNSUPPORTED}},    // API_DRIVER ANALOGUE (CUdevice_P2PAttribute)
    {"cudaComputeMode",     {"hipComputeMode",        CONV_TYPE,  API_RUNTIME, HIP_UNSUPPORTED}},    // API_DRIVER ANALOGUE (CUcomputemode)
    {"cudaFuncCache",       {"hipFuncCache_t",        CONV_CACHE, API_RUNTIME}},    // API_Driver ANALOGUE (CUfunc_cache)
    {"cudaFuncAttributes",  {"hipFuncAttributes",     CONV_EXEC,  API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaSharedMemConfig", {"hipSharedMemConfig",    CONV_TYPE,  API_RUNTIME}},
    {"cudaLimit",           {"hipLimit_t",            CONV_TYPE,  API_RUNTIME}},                     // API_Driver ANALOGUE (CUlimit)

    {"cudaOutputMode", {"hipOutputMode", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED}},

    // Texture reference management
    {"cudaTextureReadMode", {"hipTextureReadMode", CONV_TEX, API_RUNTIME}},
    {"cudaTextureFilterMode", {"hipTextureFilterMode", CONV_TEX, API_RUNTIME}},    // API_DRIVER ANALOGUE (CUfilter_mode)

    {"cudaChannelFormatKind", {"hipChannelFormatKind", CONV_TEX, API_RUNTIME}},
    {"cudaChannelFormatDesc", {"hipChannelFormatDesc", CONV_TEX, API_RUNTIME}},

    // Texture Object Management
    {"cudaResourceDesc",     {"hipResourceDesc",     CONV_TEX,     API_RUNTIME}},
    {"cudaResourceViewDesc", {"hipResourceViewDesc", CONV_TEX,     API_RUNTIME}},
    {"cudaTextureDesc",      {"hipTextureDesc",      CONV_TEX,     API_RUNTIME}},
    {"surfaceReference",     {"hipSurfaceReference", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
    // Left unchanged
    //     {"textureReference", {"textureReference", CONV_TEX, API_RUNTIME}},

    // typedefs
    {"cudaTextureObject_t",  {"hipTextureObject_t",  CONV_TEX,     API_RUNTIME}},

    // enums
    {"cudaResourceType",        {"hipResourceType",        CONV_TEX,     API_RUNTIME}},    // API_Driver ANALOGUE (CUresourcetype)
    {"cudaResourceViewFormat",  {"hipResourceViewFormat",  CONV_TEX,     API_RUNTIME}},    // API_Driver ANALOGUE (CUresourceViewFormat)
    {"cudaTextureAddressMode",  {"hipTextureAddressMode",  CONV_TEX,     API_RUNTIME}},
    {"cudaSurfaceBoundaryMode", {"hipSurfaceBoundaryMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

    {"cudaSurfaceFormatMode", {"hipSurfaceFormatMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

    // Inter-Process Communication (IPC)
    {"cudaIpcEventHandle_t",  {"hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME}},
    {"cudaIpcEventHandle_st", {"hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME}},
    {"cudaIpcMemHandle_t",    {"hipIpcMemHandle_t",   CONV_TYPE, API_RUNTIME}},
    {"cudaIpcMemHandle_st",   {"hipIpcMemHandle_t",   CONV_TYPE, API_RUNTIME}},

    // Graphics Interoperability
    {"cudaGraphicsCubeFace",      {"hipGraphicsCubeFace",      CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsMapFlags",      {"hipGraphicsMapFlags",      CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUgraphicsMapResourceFlags)
    {"cudaGraphicsRegisterFlags", {"hipGraphicsRegisterFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUgraphicsRegisterFlags)

    // OpenGL Interoperability
    {"cudaGLDeviceList", {"hipGLDeviceList", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUGLDeviceList)
    {"cudaGLMapFlags",   {"hipGLMapFlags",   CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUGLmap_flags)

    // Direct3D 9 Interoperability
    {"cudaD3D9DeviceList",    {"hipD3D9DeviceList",    CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d9DeviceList)
    {"cudaD3D9MapFlags",      {"hipD3D9MapFlags",      CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d9map_flags)
    {"cudaD3D9RegisterFlags", {"hipD3D9RegisterFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d9Register_flags)

    // Direct3D 10 Interoperability
    {"cudaD3D10DeviceList",    {"hipd3d10DeviceList",    CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d10DeviceList)
    {"cudaD3D10MapFlags",      {"hipD3D10MapFlags",      CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d10map_flags)
    {"cudaD3D10RegisterFlags", {"hipD3D10RegisterFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d10Register_flags)

    // Direct3D 11 Interoperability
    {"cudaD3D11DeviceList", {"hipd3d11DeviceList", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d11DeviceList)

    // EGL Interoperability
    {"cudaEglStreamConnection", {"hipEglStreamConnection", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUeglStreamConnection)

    ///////////////////////////// cuBLAS /////////////////////////////
    {"cublasHandle_t", {"hipblasHandle_t", CONV_TYPE, API_BLAS}},
    // TODO: dereferencing: typedef struct cublasContext *cublasHandle_t;
    //     {"cublasContext", {"hipblasHandle_t", CONV_TYPE, API_BLAS}},

    {"cublasOperation_t",   {"hipblasOperation_t",   CONV_TYPE, API_BLAS}},
    {"cublasStatus_t",      {"hipblasStatus_t",      CONV_TYPE, API_BLAS}},
    {"cublasFillMode_t",    {"hipblasFillMode_t",    CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDiagType_t",    {"hipblasDiagType_t",    CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasSideMode_t",    {"hipblasSideMode_t",    CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasPointerMode_t", {"hipblasPointerMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasAtomicsMode_t", {"hipblasAtomicsMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDataType_t",    {"hipblasDataType_t",    CONV_TYPE, API_BLAS, HIP_UNSUPPORTED}},
};

/// Maps cuda header names to hip header names.
const std::map <llvm::StringRef, hipCounter> CUDA_INCLUDE_MAP{
    // CUDA includes
    {"cuda.h",               {"hip/hip_runtime.h",        CONV_INCLUDE_CUDA_MAIN_H, API_DRIVER}},
    {"cuda_runtime.h",       {"hip/hip_runtime.h",        CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME}},
    {"cuda_runtime_api.h",   {"hip/hip_runtime_api.h",    CONV_INCLUDE,             API_RUNTIME}},
    {"channel_descriptor.h", {"hip/channel_descriptor.h", CONV_INCLUDE,             API_RUNTIME}},
    {"device_functions.h",   {"hip/device_functions.h",   CONV_INCLUDE,             API_RUNTIME}},
    {"driver_types.h",       {"hip/driver_types.h",       CONV_INCLUDE,             API_RUNTIME}},
    {"cuComplex.h",          {"hip/hip_complex.h",        CONV_INCLUDE,             API_RUNTIME}},
    {"cuda_fp16.h",          {"hip/hip_fp16.h",           CONV_INCLUDE,             API_RUNTIME}},
    {"cuda_texture_types.h", {"hip/hip_texture_types.h",  CONV_INCLUDE,             API_RUNTIME}},
    {"vector_types.h",       {"hip/hip_vector_types.h",   CONV_INCLUDE,             API_RUNTIME}},

    // CUBLAS includes
    {"cublas.h",    {"hipblas.h", CONV_INCLUDE, API_BLAS}},
    {"cublas_v2.h", {"hipblas.h", CONV_INCLUDE, API_BLAS}},

    // HIP includes
    // TODO: uncomment this when hip/cudacommon.h will be renamed to hip/hipcommon.h
    //    {"cudacommon.h", {"hipcommon.h", CONV_INCLUDE, API_RUNTIME}},
};

/// All other identifiers: function and macro names.
const std::map<llvm::StringRef, hipCounter> CUDA_IDENTIFIER_MAP{
    // Defines
    {"__CUDACC__", {"__HIPCC__", CONV_DEF, API_RUNTIME}},

    // CUDA Driver API error codes only
    {"CUDA_ERROR_INVALID_CONTEXT",               {"hipErrorInvalidContext",              CONV_TYPE, API_DRIVER}},    // 201
    {"CUDA_ERROR_CONTEXT_ALREADY_CURRENT",       {"hipErrorContextAlreadyCurrent",       CONV_TYPE, API_DRIVER}},    // 202
    {"CUDA_ERROR_ARRAY_IS_MAPPED",               {"hipErrorArrayIsMapped",               CONV_TYPE, API_DRIVER}},    // 207
    {"CUDA_ERROR_ALREADY_MAPPED",                {"hipErrorAlreadyMapped",               CONV_TYPE, API_DRIVER}},    // 208
    {"CUDA_ERROR_ALREADY_ACQUIRED",              {"hipErrorAlreadyAcquired",             CONV_TYPE, API_DRIVER}},    // 210
    {"CUDA_ERROR_NOT_MAPPED",                    {"hipErrorNotMapped",                   CONV_TYPE, API_DRIVER}},    // 211
    {"CUDA_ERROR_NOT_MAPPED_AS_ARRAY",           {"hipErrorNotMappedAsArray",            CONV_TYPE, API_DRIVER}},    // 212
    {"CUDA_ERROR_NOT_MAPPED_AS_POINTER",         {"hipErrorNotMappedAsPointer",          CONV_TYPE, API_DRIVER}},    // 213
    {"CUDA_ERROR_CONTEXT_ALREADY_IN_USE",        {"hipErrorContextAlreadyInUse",         CONV_TYPE, API_DRIVER}},    // 216
    {"CUDA_ERROR_INVALID_SOURCE",                {"hipErrorInvalidSource",               CONV_TYPE, API_DRIVER}},    // 300
    {"CUDA_ERROR_FILE_NOT_FOUND",                {"hipErrorFileNotFound",                CONV_TYPE, API_DRIVER}},    // 301
    {"CUDA_ERROR_NOT_FOUND",                     {"hipErrorNotFound",                    CONV_TYPE, API_DRIVER}},    // 500
    {"CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", {"hipErrorLaunchIncompatibleTexturing", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 703
    {"CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",        {"hipErrorPrimaryContextActive",        CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 708
    {"CUDA_ERROR_CONTEXT_IS_DESTROYED",          {"hipErrorContextIsDestroyed",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 709
    {"CUDA_ERROR_NOT_PERMITTED",                 {"hipErrorNotPermitted",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 800
    {"CUDA_ERROR_NOT_SUPPORTED",                 {"hipErrorNotSupported",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 801

    // CUDA RT API error code only
    {"cudaErrorMissingConfiguration",             {"hipErrorMissingConfiguration",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 1
    {"cudaErrorPriorLaunchFailure",               {"hipErrorPriorLaunchFailure",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 5
    {"cudaErrorInvalidDeviceFunction",            {"hipErrorInvalidDeviceFunction",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 8
    {"cudaErrorInvalidConfiguration",             {"hipErrorInvalidConfiguration",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 9
    {"cudaErrorInvalidPitchValue",                {"hipErrorInvalidPitchValue",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 12
    {"cudaErrorInvalidSymbol",                    {"hipErrorInvalidSymbol",               CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 13
    {"cudaErrorInvalidHostPointer",               {"hipErrorInvalidHostPointer",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 16
    {"cudaErrorInvalidDevicePointer",             {"hipErrorInvalidDevicePointer",        CONV_TYPE, API_RUNTIME}},    // 17
    {"cudaErrorInvalidTexture",                   {"hipErrorInvalidTexture",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 18
    {"cudaErrorInvalidTextureBinding",            {"hipErrorInvalidTextureBinding",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 19
    {"cudaErrorInvalidChannelDescriptor",         {"hipErrorInvalidChannelDescriptor",    CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 20
    {"cudaErrorInvalidMemcpyDirection",           {"hipErrorInvalidMemcpyDirection",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 21
    {"cudaErrorAddressOfConstant",                {"hipErrorAddressOfConstant",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 22
    {"cudaErrorTextureFetchFailed",               {"hipErrorTextureFetchFailed",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 23
    {"cudaErrorTextureNotBound",                  {"hipErrorTextureNotBound",             CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 24
    {"cudaErrorSynchronizationError",             {"hipErrorSynchronizationError",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 25
    {"cudaErrorInvalidFilterSetting",             {"hipErrorInvalidFilterSetting",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 26
    {"cudaErrorInvalidNormSetting",               {"hipErrorInvalidNormSetting",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 27
    {"cudaErrorMixedDeviceExecution",             {"hipErrorMixedDeviceExecution",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 28
    // Deprecated as of CUDA 4.1
    {"cudaErrorNotYetImplemented",                {"hipErrorNotYetImplemented",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 31
    // Deprecated as of CUDA 3.1
    {"cudaErrorMemoryValueTooLarge",              {"hipErrorMemoryValueTooLarge",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 32
    {"cudaErrorInsufficientDriver",               {"hipErrorInsufficientDriver",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 35
    {"cudaErrorSetOnActiveProcess",               {"hipErrorSetOnActiveProcess",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 36
    {"cudaErrorInvalidSurface",                   {"hipErrorInvalidSurface",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 37
    {"cudaErrorDuplicateVariableName",            {"hipErrorDuplicateVariableName",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 43
    {"cudaErrorDuplicateTextureName",             {"hipErrorDuplicateTextureName",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 44
    {"cudaErrorDuplicateSurfaceName",             {"hipErrorDuplicateSurfaceName",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 45
    {"cudaErrorDevicesUnavailable",               {"hipErrorDevicesUnavailable",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 46
    {"cudaErrorIncompatibleDriverContext",        {"hipErrorIncompatibleDriverContext",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 49
    {"cudaErrorDeviceAlreadyInUse",               {"hipErrorDeviceAlreadyInUse",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 54
    {"cudaErrorLaunchMaxDepthExceeded",           {"hipErrorLaunchMaxDepthExceeded",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 65
    {"cudaErrorLaunchFileScopedTex",              {"hipErrorLaunchFileScopedTex",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 66
    {"cudaErrorLaunchFileScopedSurf",             {"hipErrorLaunchFileScopedSurf",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 67
    {"cudaErrorSyncDepthExceeded",                {"hipErrorSyncDepthExceeded",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 68
    {"cudaErrorLaunchPendingCountExceeded",       {"hipErrorLaunchPendingCountExceeded",  CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 69
    {"cudaErrorNotPermitted",                     {"hipErrorNotPermitted",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 70
    {"cudaErrorNotSupported",                     {"hipErrorNotSupported",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 71
    {"cudaErrorStartupFailure",                   {"hipErrorStartupFailure",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x7f
    // Deprecated as of CUDA 4.1
    {"cudaErrorApiFailureBase",                   {"hipErrorApiFailureBase",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 10000

    {"CUDA_SUCCESS",                              {"hipSuccess",                          CONV_TYPE, API_DRIVER}},    // 0
    {"cudaSuccess",                               {"hipSuccess",                          CONV_TYPE, API_RUNTIME}},    // 0

    {"CUDA_ERROR_INVALID_VALUE",                  {"hipErrorInvalidValue",                CONV_TYPE, API_DRIVER}},    // 1
    {"cudaErrorInvalidValue",                     {"hipErrorInvalidValue",                CONV_TYPE, API_RUNTIME}},    // 11

    {"CUDA_ERROR_OUT_OF_MEMORY",                  {"hipErrorMemoryAllocation",            CONV_TYPE, API_DRIVER}},    // 2
    {"cudaErrorMemoryAllocation",                 {"hipErrorMemoryAllocation",            CONV_TYPE, API_RUNTIME}},    // 2

    {"CUDA_ERROR_NOT_INITIALIZED",                {"hipErrorNotInitialized",              CONV_TYPE, API_DRIVER}},    // 3
    {"cudaErrorInitializationError",              {"hipErrorInitializationError",         CONV_TYPE, API_RUNTIME}},    // 3

    {"CUDA_ERROR_DEINITIALIZED",                  {"hipErrorDeinitialized",               CONV_TYPE, API_DRIVER}},    // 4
    // TODO: double check, that these errors match
    {"cudaErrorCudartUnloading",                  {"hipErrorDeinitialized",               CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 29

    {"CUDA_ERROR_PROFILER_DISABLED",              {"hipErrorProfilerDisabled",            CONV_TYPE, API_DRIVER}},    // 5
    {"cudaErrorProfilerDisabled",                 {"hipErrorProfilerDisabled",            CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 55

    {"CUDA_ERROR_PROFILER_NOT_INITIALIZED",       {"hipErrorProfilerNotInitialized",      CONV_TYPE, API_DRIVER}},    // 6
    // Deprecated as of CUDA 5.0
    {"cudaErrorProfilerNotInitialized",           {"hipErrorProfilerNotInitialized",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 56

    {"CUDA_ERROR_PROFILER_ALREADY_STARTED",       {"hipErrorProfilerAlreadyStarted",      CONV_TYPE, API_DRIVER}},    // 7
    // Deprecated as of CUDA 5.0
    {"cudaErrorProfilerAlreadyStarted",           {"hipErrorProfilerAlreadyStarted",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 57

    {"CUDA_ERROR_PROFILER_ALREADY_STOPPED",       {"hipErrorProfilerAlreadyStopped",      CONV_TYPE, API_DRIVER}},    // 8
    // Deprecated as of CUDA 5.0
    {"cudaErrorProfilerAlreadyStopped",           {"hipErrorProfilerAlreadyStopped",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 58

    {"CUDA_ERROR_NO_DEVICE",                      {"hipErrorNoDevice",                    CONV_TYPE, API_DRIVER}},    // 100
    {"cudaErrorNoDevice",                         {"hipErrorNoDevice",                    CONV_TYPE, API_RUNTIME}},    // 38

    {"CUDA_ERROR_INVALID_DEVICE",                 {"hipErrorInvalidDevice",               CONV_TYPE, API_DRIVER}},    // 101
    {"cudaErrorInvalidDevice",                    {"hipErrorInvalidDevice",               CONV_TYPE, API_RUNTIME}},    // 10

    {"CUDA_ERROR_INVALID_IMAGE",                  {"hipErrorInvalidImage",                CONV_TYPE, API_DRIVER}},    // 200
    {"cudaErrorInvalidKernelImage",               {"hipErrorInvalidImage",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 47

    {"CUDA_ERROR_MAP_FAILED",                     {"hipErrorMapFailed",                   CONV_TYPE, API_DRIVER}},    // 205
    // TODO: double check, that these errors match
    {"cudaErrorMapBufferObjectFailed",            {"hipErrorMapFailed",                   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 14

    {"CUDA_ERROR_UNMAP_FAILED",                   {"hipErrorUnmapFailed",                 CONV_TYPE, API_DRIVER}},    // 206
    // TODO: double check, that these errors match
    {"cudaErrorUnmapBufferObjectFailed",          {"hipErrorUnmapFailed",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 15

    {"CUDA_ERROR_NO_BINARY_FOR_GPU",              {"hipErrorNoBinaryForGpu",              CONV_TYPE, API_DRIVER}},    // 209
    {"cudaErrorNoKernelImageForDevice",           {"hipErrorNoBinaryForGpu",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 48

    {"CUDA_ERROR_ECC_UNCORRECTABLE",              {"hipErrorECCNotCorrectable",           CONV_TYPE, API_DRIVER}},    // 214
    {"cudaErrorECCUncorrectable",                 {"hipErrorECCNotCorrectable",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 39

    {"CUDA_ERROR_UNSUPPORTED_LIMIT",              {"hipErrorUnsupportedLimit",            CONV_TYPE, API_DRIVER}},    // 215
    {"cudaErrorUnsupportedLimit",                 {"hipErrorUnsupportedLimit",            CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 42

    {"CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",        {"hipErrorPeerAccessUnsupported",       CONV_TYPE, API_DRIVER}},    // 217
    {"cudaErrorPeerAccessUnsupported",            {"hipErrorPeerAccessUnsupported",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 64

    {"CUDA_ERROR_INVALID_PTX",                    {"hipErrorInvalidKernelFile",           CONV_TYPE, API_DRIVER}},    // 218
    {"cudaErrorInvalidPtx",                       {"hipErrorInvalidKernelFile",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 78

    {"CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",       {"hipErrorInvalidGraphicsContext",      CONV_TYPE, API_DRIVER}},    // 219
    {"cudaErrorInvalidGraphicsContext",           {"hipErrorInvalidGraphicsContext",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 79

    {"CUDA_ERROR_NVLINK_UNCORRECTABLE",           {"hipErrorNvlinkUncorrectable",         CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 220
    {"cudaErrorNvlinkUncorrectable",              {"hipErrorNvlinkUncorrectable",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 80 

    {"CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", {"hipErrorSharedObjectSymbolNotFound",  CONV_TYPE, API_DRIVER}},    // 302
    {"cudaErrorSharedObjectSymbolNotFound",       {"hipErrorSharedObjectSymbolNotFound",  CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 40

    {"CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",      {"hipErrorSharedObjectInitFailed",      CONV_TYPE, API_DRIVER}},    // 303
    {"cudaErrorSharedObjectInitFailed",           {"hipErrorSharedObjectInitFailed",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 41

    {"CUDA_ERROR_OPERATING_SYSTEM",               {"hipErrorOperatingSystem",             CONV_TYPE, API_DRIVER}},    // 304
    {"cudaErrorOperatingSystem",                  {"hipErrorOperatingSystem",             CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 63

    {"CUDA_ERROR_INVALID_HANDLE",                 {"hipErrorInvalidResourceHandle",       CONV_TYPE, API_DRIVER}},    // 400
    {"cudaErrorInvalidResourceHandle",            {"hipErrorInvalidResourceHandle",       CONV_TYPE, API_RUNTIME}},    // 33

    {"CUDA_ERROR_NOT_READY",                      {"hipErrorNotReady",                    CONV_TYPE, API_DRIVER}},    // 600
    {"cudaErrorNotReady",                         {"hipErrorNotReady",                    CONV_TYPE, API_RUNTIME}},    // 34

    {"CUDA_ERROR_ILLEGAL_ADDRESS",                {"hipErrorIllegalAddress",              CONV_TYPE, API_DRIVER}},    // 700
    {"cudaErrorIllegalAddress",                   {"hipErrorIllegalAddress",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 77

    {"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",        {"hipErrorLaunchOutOfResources",        CONV_TYPE, API_DRIVER}},    // 701
    {"cudaErrorLaunchOutOfResources",             {"hipErrorLaunchOutOfResources",        CONV_TYPE, API_RUNTIME}},    // 7

    {"CUDA_ERROR_LAUNCH_TIMEOUT",                 {"hipErrorLaunchTimeOut",               CONV_TYPE, API_DRIVER}},    // 702
    {"cudaErrorLaunchTimeout",                    {"hipErrorLaunchTimeOut",               CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 6

    {"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",    {"hipErrorPeerAccessAlreadyEnabled",    CONV_TYPE, API_DRIVER}},    // 704
    {"cudaErrorPeerAccessAlreadyEnabled",         {"hipErrorPeerAccessAlreadyEnabled",    CONV_TYPE, API_RUNTIME}},    // 50

    {"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",        {"hipErrorPeerAccessNotEnabled",        CONV_TYPE, API_DRIVER}},    // 705
    {"cudaErrorPeerAccessNotEnabled",             {"hipErrorPeerAccessNotEnabled",        CONV_TYPE, API_RUNTIME}},    // 51

    {"CUDA_ERROR_ASSERT",                         {"hipErrorAssert",                      CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 710
    {"cudaErrorAssert",                           {"hipErrorAssert",                      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 59

    {"CUDA_ERROR_TOO_MANY_PEERS",                 {"hipErrorTooManyPeers",                CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 711
    {"cudaErrorTooManyPeers",                     {"hipErrorTooManyPeers",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 60

    {"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED", {"hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_DRIVER}},    // 712
    {"cudaErrorHostMemoryAlreadyRegistered",      {"hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_RUNTIME}},    // 61

    {"CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",     {"hipErrorHostMemoryNotRegistered",     CONV_TYPE, API_DRIVER}},    // 713
    {"cudaErrorHostMemoryNotRegistered",          {"hipErrorHostMemoryNotRegistered",     CONV_TYPE, API_RUNTIME}},    // 62

    {"CUDA_ERROR_HARDWARE_STACK_ERROR",           {"hipErrorHardwareStackError",          CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 714
    {"cudaErrorHardwareStackError",               {"hipErrorHardwareStackError",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 72

    {"CUDA_ERROR_ILLEGAL_INSTRUCTION",            {"hipErrorIllegalInstruction",          CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 715
    {"cudaErrorIllegalInstruction",               {"hipErrorIllegalInstruction",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 73

    {"CUDA_ERROR_MISALIGNED_ADDRESS",             {"hipErrorMisalignedAddress",           CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 716
    {"cudaErrorMisalignedAddress",                {"hipErrorMisalignedAddress",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 74

    {"CUDA_ERROR_INVALID_ADDRESS_SPACE",          {"hipErrorInvalidAddressSpace",         CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 717
    {"cudaErrorInvalidAddressSpace",              {"hipErrorInvalidAddressSpace",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 75

    {"CUDA_ERROR_INVALID_PC",                     {"hipErrorInvalidPc",                   CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 718
    {"cudaErrorInvalidPc",                        {"hipErrorInvalidPc",                   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 76

    {"CUDA_ERROR_LAUNCH_FAILED",                  {"hipErrorLaunchFailure",               CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 719
    {"cudaErrorLaunchFailure",                    {"hipErrorLaunchFailure",               CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 4

    {"CUDA_ERROR_UNKNOWN",                        {"hipErrorUnknown",                     CONV_TYPE, API_DRIVER,  HIP_UNSUPPORTED}},    // 999
    {"cudaErrorUnknown",                          {"hipErrorUnknown",                     CONV_TYPE, API_RUNTIME}},    // 30

    ///////////////////////////// CUDA DRIVER API /////////////////////////////
    // CUaddress_mode enum
    {"CU_TR_ADDRESS_MODE_WRAP",           {"HIP_TR_ADDRESS_MODE_WRAP",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0
    {"CU_TR_ADDRESS_MODE_CLAMP",          {"HIP_TR_ADDRESS_MODE_CLAMP",        CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 1
    {"CU_TR_ADDRESS_MODE_MIRROR",         {"HIP_TR_ADDRESS_MODE_MIRROR",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 2
    {"CU_TR_ADDRESS_MODE_BORDER",         {"HIP_TR_ADDRESS_MODE_BORDER",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 3

    // CUarray_cubemap_face enum
    {"CU_CUBEMAP_FACE_POSITIVE_X",        {"HIP_CUBEMAP_FACE_POSITIVE_X",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00
    {"CU_CUBEMAP_FACE_NEGATIVE_X",        {"HIP_CUBEMAP_FACE_NEGATIVE_X",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CU_CUBEMAP_FACE_POSITIVE_Y",        {"HIP_CUBEMAP_FACE_POSITIVE_Y",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02
    {"CU_CUBEMAP_FACE_NEGATIVE_Y",        {"HIP_CUBEMAP_FACE_NEGATIVE_Y",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03
    {"CU_CUBEMAP_FACE_POSITIVE_Z",        {"HIP_CUBEMAP_FACE_POSITIVE_Z",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04
    {"CU_CUBEMAP_FACE_NEGATIVE_Z",        {"HIP_CUBEMAP_FACE_NEGATIVE_Z",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x05

    // CUarray_format enum
    {"CU_AD_FORMAT_UNSIGNED_INT8",        {"HIP_AD_FORMAT_UNSIGNED_INT8",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CU_AD_FORMAT_UNSIGNED_INT16",       {"HIP_AD_FORMAT_UNSIGNED_INT16",     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02
    {"CU_AD_FORMAT_UNSIGNED_INT32",       {"HIP_AD_FORMAT_UNSIGNED_INT32",     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03
    {"CU_AD_FORMAT_SIGNED_INT8",          {"HIP_AD_FORMAT_SIGNED_INT8",        CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x08
    {"CU_AD_FORMAT_SIGNED_INT16",         {"HIP_AD_FORMAT_SIGNED_INT16",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x09
    {"CU_AD_FORMAT_SIGNED_INT32",         {"HIP_AD_FORMAT_SIGNED_INT32",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0a
    {"CU_AD_FORMAT_HALF",                 {"HIP_AD_FORMAT_HALF",               CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x10
    {"CU_AD_FORMAT_FLOAT",                {"HIP_AD_FORMAT_FLOAT",              CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x20

    // CUcomputemode enum
    {"CU_COMPUTEMODE_DEFAULT",            {"hipComputeModeDefault",            CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0 // API_RUNTIME ANALOGUE (cudaComputeModeDefault = 0)
    {"CU_COMPUTEMODE_EXCLUSIVE",          {"hipComputeModeExclusive",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 1 // API_RUNTIME ANALOGUE (cudaComputeModeExclusive = 1)
    {"CU_COMPUTEMODE_PROHIBITED",         {"hipComputeModeProhibited",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 2 // API_RUNTIME ANALOGUE (cudaComputeModeProhibited = 2)
    {"CU_COMPUTEMODE_EXCLUSIVE_PROCESS",  {"hipComputeModeExclusiveProcess",   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 3 // API_RUNTIME ANALOGUE (cudaComputeModeExclusiveProcess = 3)

    // Memory advise values
    //     {"CUmem_advise_enum", {"hipMemAdvise", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_MEM_ADVISE_SET_READ_MOSTLY",                 {"hipMemAdviseSetReadMostly",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 1 // API_RUNTIME ANALOGUE (cudaMemAdviseSetReadMostly = 1)
    {"CU_MEM_ADVISE_UNSET_READ_MOSTLY",               {"hipMemAdviseUnsetReadMostly",              CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 2 // API_RUNTIME ANALOGUE (cudaMemAdviseUnsetReadMostly = 2)
    {"CU_MEM_ADVISE_SET_PREFERRED_LOCATION",          {"hipMemAdviseSetPreferredLocation",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 3 // API_RUNTIME ANALOGUE (cudaMemAdviseSetPreferredLocation = 3)
    {"CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",        {"hipMemAdviseUnsetPreferredLocation",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 4 // API_RUNTIME ANALOGUE (cudaMemAdviseUnsetPreferredLocation = 4)
    {"CU_MEM_ADVISE_SET_ACCESSED_BY",                 {"hipMemAdviseSetAccessedBy",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 5 // API_RUNTIME ANALOGUE (cudaMemAdviseSetAccessedBy = 5)
    {"CU_MEM_ADVISE_UNSET_ACCESSED_BY",               {"hipMemAdviseUnsetAccessedBy",              CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 6 // API_RUNTIME ANALOGUE (cudaMemAdviseUnsetAccessedBy = 6)

    // CUmem_range_attribute
    //     {"CUmem_range_attribute_enum", {"hipMemRangeAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",            {"hipMemRangeAttributeReadMostly",           CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 1 // API_RUNTIME ANALOGUE (cudaMemRangeAttributeReadMostly = 1)
    {"CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",     {"hipMemRangeAttributePreferredLocation",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 2 // API_RUNTIME ANALOGUE (cudaMemRangeAttributePreferredLocation = 2)
    {"CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",            {"hipMemRangeAttributeAccessedBy",           CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 3 // API_RUNTIME ANALOGUE (cudaMemRangeAttributeAccessedBy = 3)
    {"CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION", {"hipMemRangeAttributeLastPrefetchLocation", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 4 // API_RUNTIME ANALOGUE (cudaMemRangeAttributeLastPrefetchLocation = 4)

    // CUctx_flags enum
    {"CU_CTX_SCHED_AUTO",          {"HIP_CTX_SCHED_AUTO",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00
    {"CU_CTX_SCHED_SPIN",          {"HIP_CTX_SCHED_SPIN",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CU_CTX_SCHED_YIELD",         {"HIP_CTX_SCHED_YIELD",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02
    {"CU_CTX_SCHED_BLOCKING_SYNC", {"HIP_CTX_SCHED_BLOCKING_SYNC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04
    {"CU_CTX_BLOCKING_SYNC",       {"HIP_CTX_BLOCKING_SYNC",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04
    {"CU_CTX_SCHED_MASK",          {"HIP_CTX_SCHED_MASK",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x07
    {"CU_CTX_MAP_HOST",            {"HIP_CTX_MAP_HOST",            CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x08
    {"CU_CTX_LMEM_RESIZE_TO_MAX",  {"HIP_CTX_LMEM_RESIZE_TO_MAX",  CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x10
    {"CU_CTX_FLAGS_MASK",          {"HIP_CTX_FLAGS_MASK",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1f

    // Defines
    {"CU_LAUNCH_PARAM_BUFFER_POINTER", {"HIP_LAUNCH_PARAM_BUFFER_POINTER", CONV_TYPE, API_DRIVER}},    // ((void*)0x01)
    {"CU_LAUNCH_PARAM_BUFFER_SIZE",    {"HIP_LAUNCH_PARAM_BUFFER_SIZE",    CONV_TYPE, API_DRIVER}},    // ((void*)0x02)
    {"CU_LAUNCH_PARAM_END",            {"HIP_LAUNCH_PARAM_END",            CONV_TYPE, API_DRIVER}},    // ((void*)0x00)
    {"CU_IPC_HANDLE_SIZE",             {"HIP_LAUNCH_PARAM_END",            CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 64
    {"CU_MEMHOSTALLOC_DEVICEMAP",      {"HIP_MEMHOSTALLOC_DEVICEMAP",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02
    {"CU_MEMHOSTALLOC_PORTABLE",       {"HIP_MEMHOSTALLOC_PORTABLE",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CU_MEMHOSTALLOC_WRITECOMBINED",  {"HIP_MEMHOSTALLOC_WRITECOMBINED",  CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04
    {"CU_MEMHOSTREGISTER_DEVICEMAP",   {"HIP_MEMHOSTREGISTER_DEVICEMAP",   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02
    {"CU_MEMHOSTREGISTER_IOMEMORY",    {"HIP_MEMHOSTREGISTER_IOMEMORY",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04
    {"CU_MEMHOSTREGISTER_PORTABLE",    {"HIP_MEMHOSTREGISTER_PORTABLE",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CU_PARAM_TR_DEFAULT",            {"HIP_PARAM_TR_DEFAULT",            CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // -1
    {"CU_STREAM_LEGACY",               {"HIP_STREAM_LEGACY",               CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // ((CUstream)0x1)
    {"CU_STREAM_PER_THREAD",           {"HIP_STREAM_PER_THREAD",           CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // ((CUstream)0x2)
    {"CU_TRSA_OVERRIDE_FORMAT",        {"HIP_TRSA_OVERRIDE_FORMAT",        CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CU_TRSF_NORMALIZED_COORDINATES", {"HIP_TRSF_NORMALIZED_COORDINATES", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},// 0x02
    {"CU_TRSF_READ_AS_INTEGER",        {"HIP_TRSF_READ_AS_INTEGER",        CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CU_TRSF_SRGB",                   {"HIP_TRSF_SRGB",                   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x10

    // Deprecated, use CUDA_ARRAY3D_LAYERED
    {"CUDA_ARRAY3D_2DARRAY",           {"HIP_ARRAY3D_LAYERED",             CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CUDA_ARRAY3D_CUBEMAP",           {"HIP_ARRAY3D_CUBEMAP",             CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04
    {"CUDA_ARRAY3D_DEPTH_TEXTURE",     {"HIP_ARRAY3D_DEPTH_TEXTURE",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x10
    {"CUDA_ARRAY3D_LAYERED",           {"HIP_ARRAY3D_LAYERED",             CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01
    {"CUDA_ARRAY3D_SURFACE_LDST",      {"HIP_ARRAY3D_SURFACE_LDST",        CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02
    {"CUDA_ARRAY3D_TEXTURE_GATHER",    {"HIP_ARRAY3D_TEXTURE_GATHER",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x08
    {"CUDA_VERSION",                   {"HIP_VERSION",                     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 7050

    // CUdevice_attribute_enum values...
    {"CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",                   {"hipDeviceAttributeMaxThreadsPerBlock",                   CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  1 // API_Runtime ANALOGUE (cudaDevAttrMaxThreadsPerBlock = 1)
    {"CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",                         {"hipDeviceAttributeMaxBlockDimX",                         CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  2 // API_Runtime ANALOGUE (cudaDevAttrMaxBlockDimX = 2)
    {"CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",                         {"hipDeviceAttributeMaxBlockDimY",                         CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  3 // API_Runtime ANALOGUE (cudaDevAttrMaxBlockDimY = 3)
    {"CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",                         {"hipDeviceAttributeMaxBlockDimZ",                         CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  4 // API_Runtime ANALOGUE (cudaDevAttrMaxBlockDimZ = 4)
    {"CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",                          {"hipDeviceAttributeMaxGridDimX",                          CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  5 // API_Runtime ANALOGUE (cudaDevAttrMaxGridDimX =5)
    {"CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",                          {"hipDeviceAttributeMaxGridDimY",                          CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  6 // API_Runtime ANALOGUE (cudaDevAttrMaxGridDimY = 6)
    {"CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",                          {"hipDeviceAttributeMaxGridDimZ",                          CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  7 // API_Runtime ANALOGUE (cudaDevAttrMaxGridDimZ - 7)
    {"CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",             {"hipDeviceAttributeMaxSharedMemoryPerBlock",              CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  8 // API_Runtime ANALOGUE (cudaDevAttrMaxSharedMemoryPerBlock = 8)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    {"CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK",                 {"hipDeviceAttributeMaxSharedMemoryPerBlock",              CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  8
    {"CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",                   {"hipDeviceAttributeTotalConstantMemory",                  CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    //  9 // API_Runtime ANALOGUE (cudaDevAttrTotalConstantMemory = 9)
    {"CU_DEVICE_ATTRIBUTE_WARP_SIZE",                               {"hipDeviceAttributeWarpSize",                             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 10 // API_Runtime ANALOGUE (cudaDevAttrWarpSize = 10)
    {"CU_DEVICE_ATTRIBUTE_MAX_PITCH",                               {"hipDeviceAttributeMaxPitch",                             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 11 // API_Runtime ANALOGUE (cudaDevAttrMaxPitch = 11)
    {"CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",                 {"hipDeviceAttributeMaxRegistersPerBlock",                 CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 12 // API_Runtime ANALOGUE (cudaDevAttrMaxRegistersPerBlock = 12)
    {"CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK",                     {"hipDeviceAttributeMaxRegistersPerBlock",                 CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 12
    {"CU_DEVICE_ATTRIBUTE_CLOCK_RATE",                              {"hipDeviceAttributeClockRate",                            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 13 // API_Runtime ANALOGUE (cudaDevAttrMaxRegistersPerBlock = 13)
    {"CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",                       {"hipDeviceAttributeTextureAlignment",                     CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 14 // API_Runtime ANALOGUE (cudaDevAttrTextureAlignment = 14)
    // Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
    {"CU_DEVICE_ATTRIBUTE_GPU_OVERLAP",                             {"hipDeviceAttributeAsyncEngineCount",                     CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 15 // API_Runtime ANALOGUE (cudaDevAttrGpuOverlap = 15)
    {"CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",                    {"hipDeviceAttributeMultiprocessorCount",                  CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 16 // API_Runtime ANALOGUE (cudaDevAttrMultiProcessorCount = 16)
    {"CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",                     {"hipDeviceAttributeKernelExecTimeout",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 17 // API_Runtime ANALOGUE (cudaDevAttrKernelExecTimeout = 17)
    {"CU_DEVICE_ATTRIBUTE_INTEGRATED",                              {"hipDeviceAttributeIntegrated",                           CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 18 // API_Runtime ANALOGUE (cudaDevAttrIntegrated = 18)
    {"CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",                     {"hipDeviceAttributeCanMapHostMemory",                     CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 19 // API_Runtime ANALOGUE (cudaDevAttrCanMapHostMemory = 19)
    {"CU_DEVICE_ATTRIBUTE_COMPUTE_MODE",                            {"hipDeviceAttributeComputeMode",                          CONV_TYPE,                API_DRIVER}},                      // 20 // API_Runtime ANALOGUE (cudaDevAttrComputeMode = 20)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH",                 {"hipDeviceAttributeMaxTexture1DWidth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 21 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DWidth = 21)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH",                 {"hipDeviceAttributeMaxTexture2DWidth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 22 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DWidth = 22)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT",                {"hipDeviceAttributeMaxTexture2DHeight",                   CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 23 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DHeight = 23)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH",                 {"hipDeviceAttributeMaxTexture3DWidth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 24 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DWidth = 24)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT",                {"hipDeviceAttributeMaxTexture3DHeight",                   CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 25 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DHeight = 25)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH",                 {"hipDeviceAttributeMaxTexture3DDepth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 26 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DDepth = 26)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH",         {"hipDeviceAttributeMaxTexture2DLayeredWidth",             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 27 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredWidth = 27)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT",        {"hipDeviceAttributeMaxTexture2DLayeredHeight",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 28 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredHeight = 28)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS",        {"hipDeviceAttributeMaxTexture2DLayeredLayers",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 29 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredLayers = 29)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH",           {"hipDeviceAttributeMaxTexture2DLayeredWidth",             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 27 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredWidth = 27)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT",          {"hipDeviceAttributeMaxTexture2DLayeredHeight",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 28 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredHeight = 28)
    // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES",       {"hipDeviceAttributeMaxTexture2DLayeredLayers",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 29 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLayeredLayers = 29)
    {"CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",                       {"hipDeviceAttributeSurfaceAlignment",                     CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 30 // API_Runtime ANALOGUE (cudaDevAttrSurfaceAlignment = 30)
    {"CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",                      {"hipDeviceAttributeConcurrentKernels",                    CONV_TYPE,                API_DRIVER}},                      // 31 // API_Runtime ANALOGUE (cudaDevAttrConcurrentKernels = 31)
    {"CU_DEVICE_ATTRIBUTE_ECC_ENABLED",                             {"hipDeviceAttributeEccEnabled",                           CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 32 // API_Runtime ANALOGUE (cudaDevAttrEccEnabled = 32)
    {"CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",                              {"hipDeviceAttributePciBusId",                             CONV_TYPE,                API_DRIVER}},                      // 33 // API_Runtime ANALOGUE (cudaDevAttrPciBusId = 33)
    {"CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",                           {"hipDeviceAttributePciDeviceId",                          CONV_TYPE,                API_DRIVER}},                      // 34 // API_Runtime ANALOGUE (cudaDevAttrPciDeviceId = 34)
    {"CU_DEVICE_ATTRIBUTE_TCC_DRIVER",                              {"hipDeviceAttributeTccDriver",                            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 35 // API_Runtime ANALOGUE (cudaDevAttrTccDriver = 35)
    {"CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE",                       {"hipDeviceAttributeMemoryClockRate",                      CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 36 // API_Runtime ANALOGUE (cudaDevAttrMemoryClockRate = 36)
    {"CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",                 {"hipDeviceAttributeMemoryBusWidth",                       CONV_TYPE,                API_DRIVER}},                      // 37 // API_Runtime ANALOGUE (cudaDevAttrGlobalMemoryBusWidth = 37)
    {"CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",                           {"hipDeviceAttributeL2CacheSize",                          CONV_TYPE,                API_DRIVER}},                      // 38 // API_Runtime ANALOGUE (cudaDevAttrL2CacheSize = 38)
    {"CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",          {"hipDeviceAttributeMaxThreadsPerMultiProcessor",          CONV_TYPE,                API_DRIVER}},                      // 39 // API_Runtime ANALOGUE (cudaDevAttrMaxThreadsPerMultiProcessor = 39)
    {"CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",                      {"hipDeviceAttributeAsyncEngineCount",                     CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 40 // API_Runtime ANALOGUE (cudaDevAttrAsyncEngineCount = 40)
    {"CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",                      {"hipDeviceAttributeUnifiedAddressing",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 41 // API_Runtime ANALOGUE (cudaDevAttrUnifiedAddressing = 41)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH",         {"hipDeviceAttributeMaxTexture1DLayeredWidth",             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 42 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DLayeredWidth = 42)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS",        {"hipDeviceAttributeMaxTexture1DLayeredLayers",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 43 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DLayeredLayers = 43)
    // deprecated, do not use
    {"CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER",                        {"hipDeviceAttributeCanTex2DGather",                       CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 44 // API_Runtime ANALOGUE (no)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH",          {"hipDeviceAttributeMaxTexture2DGatherWidth",              CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 45 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DGatherWidth = 45)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT",         {"hipDeviceAttributeMaxTexture2DGatherHeight",             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 46 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DGatherHeight = 46)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE",       {"hipDeviceAttributeMaxTexture3DWidthAlternate",           CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 47 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DWidthAlt = 47)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE",      {"hipDeviceAttributeMaxTexture3DHeightAlternate",          CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 48 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DHeightAlt = 48)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE",       {"hipDeviceAttributeMaxTexture3DDepthAlternate",           CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 49 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture3DDepthAlt = 49)
    {"CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",                           {"hipDeviceAttributePciDomainId",                          CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 50 // API_Runtime ANALOGUE (cudaDevAttrPciDomainId = 50)
    {"CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",                 {"hipDeviceAttributeTexturePitchAlignment",                CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 51 // API_Runtime ANALOGUE (cudaDevAttrTexturePitchAlignment = 51)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH",            {"hipDeviceAttributeMaxTextureCubemapWidth",               CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 52 // API_Runtime ANALOGUE (cudaDevAttrMaxTextureCubemapWidth = 52)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH",    {"hipDeviceAttributeMaxTextureCubemapLayeredWidth",        CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 53 // API_Runtime ANALOGUE (cudaDevAttrMaxTextureCubemapLayeredWidth = 53)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS",   {"hipDeviceAttributeMaxTextureCubemapLayeredLayers",       CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 54 // API_Runtime ANALOGUE (cudaDevAttrMaxTextureCubemapLayeredLayers = 54)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH",                 {"hipDeviceAttributeMaxSurface1DWidth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 55 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface1DWidth = 55)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH",                 {"hipDeviceAttributeMaxSurface2DWidth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 56 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DWidth = 56)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT",                {"hipDeviceAttributeMaxSurface2DHeight",                   CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 57 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DHeight = 57)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH",                 {"hipDeviceAttributeMaxSurface3DWidth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 58 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface3DWidth = 58)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT",                {"hipDeviceAttributeMaxSurface3DHeight",                   CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 59 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface3DHeight = 59)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH",                 {"hipDeviceAttributeMaxSurface3DDepth",                    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 60 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface3DDepth = 60)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH",         {"hipDeviceAttributeMaxSurface1DLayeredWidth",             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 61 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface1DLayeredWidth = 61)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS",        {"hipDeviceAttributeMaxSurface1DLayeredLayers",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 62 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface1DLayeredLayers = 62)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH",         {"hipDeviceAttributeMaxSurface2DLayeredWidth",             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 63 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DLayeredWidth = 63)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT",        {"hipDeviceAttributeMaxSurface2DLayeredHeight",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 64 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DLayeredHeight = 64)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS",        {"hipDeviceAttributeMaxSurface2DLayeredLayers",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 65 // API_Runtime ANALOGUE (cudaDevAttrMaxSurface2DLayeredLayers = 65)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH",            {"hipDeviceAttributeMaxSurfaceCubemapWidth",               CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 66 // API_Runtime ANALOGUE (cudaDevAttrMaxSurfaceCubemapWidth = 66)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH",    {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",        CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 67 // API_Runtime ANALOGUE (cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS",   {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",       CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 68 // API_Runtime ANALOGUE (cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH",          {"hipDeviceAttributeMaxTexture1DLinearWidth",              CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 69 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DLinearWidth = 69)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH",          {"hipDeviceAttributeMaxTexture2DLinearWidth",              CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 70 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLinearWidth = 70)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT",         {"hipDeviceAttributeMaxTexture2DLinearHeight",             CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 71 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLinearHeight = 71)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH",          {"hipDeviceAttributeMaxTexture2DLinearPitch",              CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 72 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DLinearPitch = 72)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH",       {"hipDeviceAttributeMaxTexture2DMipmappedWidth",           CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 73 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DMipmappedWidth = 73)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT",      {"hipDeviceAttributeMaxTexture2DMipmappedHeight",          CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 74 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture2DMipmappedHeight = 74)
    {"CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",                {"hipDeviceAttributeComputeCapabilityMajor",               CONV_TYPE,                API_DRIVER}},                      // 75 // API_Runtime ANALOGUE (cudaDevAttrComputeCapabilityMajor = 75)
    {"CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",                {"hipDeviceAttributeComputeCapabilityMinor",               CONV_TYPE,                API_DRIVER}},                      // 76 // API_Runtime ANALOGUE (cudaDevAttrComputeCapabilityMinor = 76)
    {"CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH",       {"hipDeviceAttributeMaxTexture1DMipmappedWidth",           CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 77 // API_Runtime ANALOGUE (cudaDevAttrMaxTexture1DMipmappedWidth = 77)
    {"CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",             {"hipDeviceAttributeStreamPrioritiesSupported",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 78 // API_Runtime ANALOGUE (cudaDevAttrStreamPrioritiesSupported = 78)
    {"CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",               {"hipDeviceAttributeGlobalL1CacheSupported",               CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 79 // API_Runtime ANALOGUE (cudaDevAttrGlobalL1CacheSupported = 79)
    {"CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED",                {"hipDeviceAttributeLocalL1CacheSupported",                CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 80 // API_Runtime ANALOGUE (cudaDevAttrLocalL1CacheSupported = 80)
    {"CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",    {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",     CONV_TYPE,                API_DRIVER}},                      // 81 // API_Runtime ANALOGUE (cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81)
    {"CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",        {"hipDeviceAttributeMaxRegistersPerMultiprocessor",        CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 82 // API_Runtime ANALOGUE (cudaDevAttrMaxRegistersPerMultiprocessor = 82)
    {"CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY",                          {"hipDeviceAttributeManagedMemory",                        CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 83 // API_Runtime ANALOGUE (cudaDevAttrManagedMemory = 83)
    {"CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",                         {"hipDeviceAttributeIsMultiGpuBoard",                      CONV_TYPE,                API_DRIVER}},                      // 84 // API_Runtime ANALOGUE (cudaDevAttrIsMultiGpuBoard = 84)
    {"CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID",                {"hipDeviceAttributeMultiGpuBoardGroupId",                 CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 85 // API_Runtime ANALOGUE (cudaDevAttrMultiGpuBoardGroupID = 85)
    {"CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",            {"hipDeviceAttributeHostNativeAtomicSupported",            CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 86 // API_Runtime ANALOGUE (cudaDevAttrHostNativeAtomicSupported = 86)
    {"CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",   {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio",     CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 87 // API_Runtime ANALOGUE (cudaDevAttrSingleToDoublePrecisionPerfRatio = 87)
    {"CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",                  {"hipDeviceAttributePageableMemoryAccess",                 CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 88 // API_Runtime ANALOGUE (cudaDevAttrPageableMemoryAccess = 88)
    {"CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",               {"hipDeviceAttributeConcurrentManagedAccess",              CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 89 // API_Runtime ANALOGUE (cudaDevAttrConcurrentManagedAccess = 89)
    {"CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED",            {"hipDeviceAttributeComputePreemptionSupported",           CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 90 // API_Runtime ANALOGUE (cudaDevAttrComputePreemptionSupported = 90)
    {"CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM", {"hipDeviceAttributeCanUseHostPointerForRegisteredMem",    CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 91 // API_Runtime ANALOGUE (cudaDevAttrCanUseHostPointerForRegisteredMem = 91)

    {"CU_DEVICE_ATTRIBUTE_MAX",                                     {"hipDeviceAttributeMax",                                  CONV_TYPE,                API_DRIVER,  HIP_UNSUPPORTED}},    // 92 // API_Runtime ANALOGUE (no)

    // TODO: Analogous enum is needed in HIP. Couldn't map enum to struct hipPointerAttribute_t.
    // TODO: Do for Pointer Attributes the same as for Device Attributes.
    //     {"CUpointer_attribute_enum", {"hipPointerAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    //     {"CUpointer_attribute", {"hipPointerAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_CONTEXT",        {"hipPointerAttributeContext",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 1 // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_MEMORY_TYPE",    {"hipPointerAttributeMemoryType",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 2 // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_DEVICE_POINTER", {"hipPointerAttributeDevicePointer", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 3 // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_HOST_POINTER",   {"hipPointerAttributeHostPointer",   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 4 // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_P2P_TOKENS",     {"hipPointerAttributeP2pTokens",     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 5 // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_SYNC_MEMOPS",    {"hipPointerAttributeSyncMemops",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 6 // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_BUFFER_ID",      {"hipPointerAttributeBufferId",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 7 // API_Runtime ANALOGUE (no)
    {"CU_POINTER_ATTRIBUTE_IS_MANAGED",     {"hipPointerAttributeIsManaged",     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 8 // API_Runtime ANALOGUE (no)

    // CUfunction_attribute_enum values
    {"CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK", {"hipFuncAttributeMaxThreadsPerBlocks", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",     {"hipFuncAttributeSharedSizeBytes",     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",      {"hipFuncAttributeConstSizeBytes",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",      {"hipFuncAttributeLocalSizeBytes",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_NUM_REGS",              {"hipFuncAttributeNumRegs",             CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_PTX_VERSION",           {"hipFuncAttributePtxVersion",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_BINARY_VERSION",        {"hipFuncAttributeBinaryVersion",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_CACHE_MODE_CA",         {"hipFuncAttributeCacheModeCA",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_FUNC_ATTRIBUTE_MAX",                   {"hipFuncAttributeMax",                 CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},

    // enum CUgraphicsMapResourceFlags/CUgraphicsMapResourceFlags_enum
    {"CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE",          {"hipGraphicsMapFlagsNone",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaGraphicsMapFlagsNone = 0)
    {"CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY",     {"hipGraphicsMapFlagsReadOnly",     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaGraphicsMapFlagsReadOnly = 1)
    {"CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD", {"hipGraphicsMapFlagsWriteDiscard", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaGraphicsMapFlagsWriteDiscard = 2)

    // enum CUgraphicsRegisterFlags/CUgraphicsRegisterFlags_enum
    {"CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE",       {"hipGraphicsRegisterFlagsNone",             CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsNone = 0)
    {"CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY",  {"hipGraphicsRegisterFlagsReadOnly",         CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsReadOnly = 1)
    {"CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD",  {"hipGraphicsRegisterFlagsWriteDiscard",     CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsWriteDiscard = 2)
    {"CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST",   {"hipGraphicsRegisterFlagsSurfaceLoadStore", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsSurfaceLoadStore = 4)
    {"CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER", {"hipGraphicsRegisterFlagsTextureGather",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x08 // API_Runtime ANALOGUE (cudaGraphicsRegisterFlagsTextureGather = 8)

    // enum CUoccupancy_flags/CUoccupancy_flags_enum
    {"CU_OCCUPANCY_DEFAULT",                  {"hipOccupancyDefault",                CONV_TYPE,  API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaOccupancyDefault = 0x0)
    {"CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE", {"hipOccupancyDisableCachingOverride", CONV_TYPE,  API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaOccupancyDisableCachingOverride = 0x1)

    // enum CUfunc_cache/CUfunc_cache_enum
    {"CU_FUNC_CACHE_PREFER_NONE",             {"hipFuncCachePreferNone",             CONV_CACHE, API_DRIVER}},    // 0x00 // API_Runtime ANALOGUE (cudaFilterModePoint = 0)
    {"CU_FUNC_CACHE_PREFER_SHARED",           {"hipFuncCachePreferShared",           CONV_CACHE, API_DRIVER}},    // 0x01 // API_Runtime ANALOGUE (cudaFuncCachePreferShared = 1)
    {"CU_FUNC_CACHE_PREFER_L1",               {"hipFuncCachePreferL1",               CONV_CACHE, API_DRIVER}},    // 0x02 // API_Runtime ANALOGUE (cudaFuncCachePreferL1 = 2)
    {"CU_FUNC_CACHE_PREFER_EQUAL",            {"hipFuncCachePreferEqual",            CONV_CACHE, API_DRIVER}},    // 0x03 // API_Runtime ANALOGUE (cudaFuncCachePreferEqual = 3)

    // enum CUipcMem_flags/CUipcMem_flags_enum
    {"CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS", {"hipIpcMemLazyEnablePeerAccess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1 // API_Runtime ANALOGUE (cudaIpcMemLazyEnablePeerAccess = 0x01)

    // JIT
    // enum CUjit_cacheMode/CUjit_cacheMode_enum
    {"CU_JIT_CACHE_OPTION_NONE",           {"hipJitCacheModeOptionNone",           CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_JIT_CACHE_OPTION_CG",             {"hipJitCacheModeOptionCG",             CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_JIT_CACHE_OPTION_CA",             {"hipJitCacheModeOptionCA",             CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    // enum CUjit_fallback/CUjit_fallback_enum
    {"CU_PREFER_PTX",                      {"hipJitFallbackPreferPtx",             CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_PREFER_BINARY",                   {"hipJitFallbackPreferBinary",          CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    // enum CUjit_option/CUjit_option_enum
    {"CU_JIT_MAX_REGISTERS",               {"hipJitOptionMaxRegisters",            CONV_JIT, API_DRIVER}},
    {"CU_JIT_THREADS_PER_BLOCK",           {"hipJitOptionThreadsPerBlock",         CONV_JIT, API_DRIVER}},
    {"CU_JIT_WALL_TIME",                   {"hipJitOptionWallTime",                CONV_JIT, API_DRIVER}},
    {"CU_JIT_INFO_LOG_BUFFER",             {"hipJitOptionInfoLogBuffer",           CONV_JIT, API_DRIVER}},
    {"CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES",  {"hipJitOptionInfoLogBufferSizeBytes",  CONV_JIT, API_DRIVER}},
    {"CU_JIT_ERROR_LOG_BUFFER",            {"hipJitOptionErrorLogBuffer",          CONV_JIT, API_DRIVER}},
    {"CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES", {"hipJitOptionErrorLogBufferSizeBytes", CONV_JIT, API_DRIVER}},
    {"CU_JIT_OPTIMIZATION_LEVEL",          {"hipJitOptionOptimizationLevel",       CONV_JIT, API_DRIVER}},
    {"CU_JIT_TARGET_FROM_CUCONTEXT",       {"hipJitOptionTargetFromContext",       CONV_JIT, API_DRIVER}},
    {"CU_JIT_TARGET",                      {"hipJitOptionTarget",                  CONV_JIT, API_DRIVER}},
    {"CU_JIT_FALLBACK_STRATEGY",           {"hipJitOptionFallbackStrategy",        CONV_JIT, API_DRIVER}},
    {"CU_JIT_GENERATE_DEBUG_INFO",         {"hipJitOptionGenerateDebugInfo",       CONV_JIT, API_DRIVER}},
    {"CU_JIT_LOG_VERBOSE",                 {"hipJitOptionLogVerbose",              CONV_JIT, API_DRIVER}},
    {"CU_JIT_GENERATE_LINE_INFO",          {"hipJitOptionGenerateLineInfo",        CONV_JIT, API_DRIVER}},
    {"CU_JIT_CACHE_MODE",                  {"hipJitOptionCacheMode",               CONV_JIT, API_DRIVER}},
    {"CU_JIT_NEW_SM3X_OPT",                {"hipJitOptionSm3xOpt",                 CONV_JIT, API_DRIVER}},
    {"CU_JIT_FAST_COMPILE",                {"hipJitOptionFastCompile",             CONV_JIT, API_DRIVER}},
    {"CU_JIT_NUM_OPTIONS",                 {"hipJitOptionNumOptions",              CONV_JIT, API_DRIVER}},

    // enum CUjit_target/CUjit_target_enum
    {"CU_TARGET_COMPUTE_10",               {"hipJitTargetCompute10",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_11",               {"hipJitTargetCompute11",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_12",               {"hipJitTargetCompute12",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_13",               {"hipJitTargetCompute13",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_20",               {"hipJitTargetCompute20",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_21",               {"hipJitTargetCompute21",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_30",               {"hipJitTargetCompute30",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_32",               {"hipJitTargetCompute32",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_35",               {"hipJitTargetCompute35",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_37",               {"hipJitTargetCompute37",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_50",               {"hipJitTargetCompute50",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_52",               {"hipJitTargetCompute52",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_53",               {"hipJitTargetCompute53",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_60",               {"hipJitTargetCompute60",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_61",               {"hipJitTargetCompute61",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_TARGET_COMPUTE_62",               {"hipJitTargetCompute62",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    // enum CUjitInputType/CUjitInputType_enum
    {"CU_JIT_INPUT_CUBIN",                 {"hipJitInputTypeBin",                  CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_JIT_INPUT_PTX",                   {"hipJitInputTypePtx",                  CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_JIT_INPUT_FATBINARY",             {"hipJitInputTypeFatBinary",            CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_JIT_INPUT_OBJECT",                {"hipJitInputTypeObject",               CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_JIT_INPUT_LIBRARY",               {"hipJitInputTypeLibrary",              CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},
    {"CU_JIT_NUM_INPUT_TYPES",             {"hipJitInputTypeNumInputTypes",        CONV_JIT, API_DRIVER, HIP_UNSUPPORTED}},

    // enum CUlimit/CUlimit_enum
    {"CU_LIMIT_STACK_SIZE",                       {"hipLimitStackSize",                    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaLimitStackSize = 0x00)
    {"CU_LIMIT_PRINTF_FIFO_SIZE",                 {"hipLimitPrintfFifoSize",               CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaLimitPrintfFifoSize = 0x01)
    {"CU_LIMIT_MALLOC_HEAP_SIZE",                 {"hipLimitMallocHeapSize",               CONV_TYPE, API_DRIVER}},    // 0x02 // API_Runtime ANALOGUE (cudaLimitMallocHeapSize = 0x02)
    {"CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH",           {"hipLimitDevRuntimeSyncDepth",          CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaLimitDevRuntimeSyncDepth = 0x03)
    {"CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT", {"hipLimitDevRuntimePendingLaunchCount", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04 // API_Runtime ANALOGUE (cudaLimitDevRuntimePendingLaunchCount = 0x04)
    {"CU_LIMIT_STACK_SIZE",                       {"hipLimitStackSize",                    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (no)

    // enum CUmemAttach_flags/CUmemAttach_flags_enum
    {"CU_MEM_ATTACH_GLOBAL",   {"hipMemAttachGlobal",  CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1 // API_Runtime ANALOGUE (#define cudaMemAttachGlobal 0x01)
    {"CU_MEM_ATTACH_HOST",     {"hipMemAttachHost",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x2 // API_Runtime ANALOGUE (#define cudaMemAttachHost 0x02)
    {"CU_MEM_ATTACH_SINGLE",   {"hipMemAttachSingle",  CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x4 // API_Runtime ANALOGUE (#define cudaMemAttachSingle 0x04)

    // enum CUmemorytype/CUmemorytype_enum
    {"CU_MEMORYTYPE_HOST",    {"hipMemTypeHost",    CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (no)
    {"CU_MEMORYTYPE_DEVICE",  {"hipMemTypeDevice",  CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (no)
    {"CU_MEMORYTYPE_ARRAY",   {"hipMemTypeArray",   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (no)
    {"CU_MEMORYTYPE_UNIFIED", {"hipMemTypeUnified", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04 // API_Runtime ANALOGUE (no)

    // enum CUresourcetype/CUresourcetype_enum
    {"CU_RESOURCE_TYPE_ARRAY",           {"hipResourceTypeArray",          CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaResourceTypeArray = 0x00)
    {"CU_RESOURCE_TYPE_MIPMAPPED_ARRAY", {"hipResourceTypeMipmappedArray", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaResourceTypeMipmappedArray = 0x01)
    {"CU_RESOURCE_TYPE_LINEAR",          {"hipResourceTypeLinear",         CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaResourceTypeLinear = 0x02)
    {"CU_RESOURCE_TYPE_PITCH2D",         {"hipResourceTypePitch2D",        CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaResourceTypePitch2D = 0x03)

    // enum CUresourceViewFormat/CUresourceViewFormat_enum
    {"CU_RES_VIEW_FORMAT_NONE",          {"hipResViewFormatNone",                      CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaResViewFormatNone = 0x00)
    {"CU_RES_VIEW_FORMAT_UINT_1X8",      {"hipResViewFormatUnsignedChar1",             CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedChar1 = 0x01)
    {"CU_RES_VIEW_FORMAT_UINT_2X8",      {"hipResViewFormatUnsignedChar2",             CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedChar2 = 0x02)
    {"CU_RES_VIEW_FORMAT_UINT_4X8",      {"hipResViewFormatUnsignedChar4",             CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedChar4 = 0x03)
    {"CU_RES_VIEW_FORMAT_SINT_1X8",      {"hipResViewFormatSignedChar1",               CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x04 // API_Runtime ANALOGUE (cudaResViewFormatSignedChar1 = 0x04)
    {"CU_RES_VIEW_FORMAT_SINT_2X8",      {"hipResViewFormatSignedChar2",               CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x05 // API_Runtime ANALOGUE (cudaResViewFormatSignedChar2 = 0x05)
    {"CU_RES_VIEW_FORMAT_SINT_4X8",      {"hipResViewFormatSignedChar4",               CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x06 // API_Runtime ANALOGUE (cudaResViewFormatSignedChar4 = 0x06)
    {"CU_RES_VIEW_FORMAT_UINT_1X16",     {"hipResViewFormatUnsignedShort1",            CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x07 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedShort1 = 0x07)
    {"CU_RES_VIEW_FORMAT_UINT_2X16",     {"hipResViewFormatUnsignedShort2",            CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x08 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedShort2 = 0x08)
    {"CU_RES_VIEW_FORMAT_UINT_4X16",     {"hipResViewFormatUnsignedShort4",            CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x09 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedShort4 = 0x09)
    {"CU_RES_VIEW_FORMAT_SINT_1X16",     {"hipResViewFormatSignedShort1",              CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0a // API_Runtime ANALOGUE (cudaResViewFormatSignedShort1 = 0x0a)
    {"CU_RES_VIEW_FORMAT_SINT_2X16",     {"hipResViewFormatSignedShort2",              CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0b // API_Runtime ANALOGUE (cudaResViewFormatSignedShort2 = 0x0b)
    {"CU_RES_VIEW_FORMAT_SINT_4X16",     {"hipResViewFormatSignedShort4",              CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0c // API_Runtime ANALOGUE (cudaResViewFormatSignedShort4 = 0x0c)
    {"CU_RES_VIEW_FORMAT_UINT_1X32",     {"hipResViewFormatUnsignedInt1",              CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0d // API_Runtime ANALOGUE (cudaResViewFormatUnsignedInt1 = 0x0d)
    {"CU_RES_VIEW_FORMAT_UINT_2X32",     {"hipResViewFormatUnsignedInt2",              CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0e // API_Runtime ANALOGUE (cudaResViewFormatUnsignedInt2 = 0x0e)
    {"CU_RES_VIEW_FORMAT_UINT_4X32",     {"hipResViewFormatUnsignedInt4",              CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0f // API_Runtime ANALOGUE (cudaResViewFormatUnsignedInt4 = 0x0f)
    {"CU_RES_VIEW_FORMAT_SINT_1X32",     {"hipResViewFormatSignedInt1",                CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x10 // API_Runtime ANALOGUE (cudaResViewFormatSignedInt1 = 0x10)
    {"CU_RES_VIEW_FORMAT_SINT_2X32",     {"hipResViewFormatSignedInt2",                CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x11 // API_Runtime ANALOGUE (cudaResViewFormatSignedInt2 = 0x11)
    {"CU_RES_VIEW_FORMAT_SINT_4X32",     {"hipResViewFormatSignedInt4",                CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x12 // API_Runtime ANALOGUE (cudaResViewFormatSignedInt4 = 0x12)
    {"CU_RES_VIEW_FORMAT_FLOAT_1X16",    {"hipResViewFormatHalf1",                     CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x13 // API_Runtime ANALOGUE (cudaResViewFormatHalf1 = 0x13)
    {"CU_RES_VIEW_FORMAT_FLOAT_2X16",    {"hipResViewFormatHalf2",                     CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x14 // API_Runtime ANALOGUE (cudaResViewFormatHalf2 = 0x14)
    {"CU_RES_VIEW_FORMAT_FLOAT_4X16",    {"hipResViewFormatHalf4",                     CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x15 // API_Runtime ANALOGUE (cudaResViewFormatHalf4 = 0x15)
    {"CU_RES_VIEW_FORMAT_FLOAT_1X32",    {"hipResViewFormatFloat1",                    CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x16 // API_Runtime ANALOGUE (cudaResViewFormatFloat1 = 0x16)
    {"CU_RES_VIEW_FORMAT_FLOAT_2X32",    {"hipResViewFormatFloat2",                    CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x17 // API_Runtime ANALOGUE (cudaResViewFormatFloat2 = 0x17)
    {"CU_RES_VIEW_FORMAT_FLOAT_4X32",    {"hipResViewFormatFloat4",                    CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x18 // API_Runtime ANALOGUE (cudaResViewFormatFloat4 = 0x18)
    {"CU_RES_VIEW_FORMAT_UNSIGNED_BC1",  {"hipResViewFormatUnsignedBlockCompressed1",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x19 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed1 = 0x19)
    {"CU_RES_VIEW_FORMAT_UNSIGNED_BC2",  {"hipResViewFormatUnsignedBlockCompressed2",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1a // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed2 = 0x1a)
    {"CU_RES_VIEW_FORMAT_UNSIGNED_BC3",  {"hipResViewFormatUnsignedBlockCompressed3",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1b // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed3 = 0x1b)
    {"CU_RES_VIEW_FORMAT_UNSIGNED_BC4",  {"hipResViewFormatUnsignedBlockCompressed4",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1c // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed4 = 0x1c)
    {"CU_RES_VIEW_FORMAT_SIGNED_BC4",    {"hipResViewFormatSignedBlockCompressed4",    CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1d // API_Runtime ANALOGUE (cudaResViewFormatSignedBlockCompressed4 = 0x1d)
    {"CU_RES_VIEW_FORMAT_UNSIGNED_BC5",  {"hipResViewFormatUnsignedBlockCompressed5",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1e // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed5 = 0x1e)
    {"CU_RES_VIEW_FORMAT_SIGNED_BC5",    {"hipResViewFormatSignedBlockCompressed5",    CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1f // API_Runtime ANALOGUE (cudaResViewFormatSignedBlockCompressed5 = 0x1f)
    {"CU_RES_VIEW_FORMAT_UNSIGNED_BC6H", {"hipResViewFormatUnsignedBlockCompressed6H", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x20 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed6H = 0x20)
    {"CU_RES_VIEW_FORMAT_SIGNED_BC6H",   {"hipResViewFormatSignedBlockCompressed6H",   CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x21 // API_Runtime ANALOGUE (cudaResViewFormatSignedBlockCompressed6H = 0x21)
    {"CU_RES_VIEW_FORMAT_UNSIGNED_BC7",  {"hipResViewFormatUnsignedBlockCompressed7",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 0x22 // API_Runtime ANALOGUE (cudaResViewFormatUnsignedBlockCompressed7 = 0x22)

    {"CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE",    {"hipSharedMemBankSizeDefault",   CONV_TYPE, API_DRIVER}},
    {"CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE",  {"hipSharedMemBankSizeFourByte",  CONV_TYPE, API_DRIVER}},
    {"CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE", {"hipSharedMemBankSizeEightByte", CONV_TYPE, API_DRIVER}},

    // enum CUstream_flags/CUstream_flags_enum
    {"CU_STREAM_DEFAULT",      {"hipStreamDefault",      CONV_TYPE, API_DRIVER}},
    {"CU_STREAM_NON_BLOCKING", {"hipStreamNonBlocking",  CONV_TYPE, API_DRIVER}},

    // Flags for ::cuStreamWaitValue32 (enum CUstreamWaitValue_flags/CUstreamWaitValue_flags_enum)
    {"CU_STREAM_WAIT_VALUE_GEQ",                {"hipStreamWaitValueGeq",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0
    {"CU_STREAM_WAIT_VALUE_EQ",                 {"hipStreamWaitValueEq",                 CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1
    {"CU_STREAM_WAIT_VALUE_AND",                {"hipStreamWaitValueAnd",                CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x2
    {"CU_STREAM_WAIT_VALUE_FLUSH",              {"hipStreamWaitValueFlush",              CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 1<<30

    // Flags for ::cuStreamWriteValue32 (enum CUstreamWriteValue_flags/CUstreamWriteValue_flags_enum)
    {"CU_STREAM_WRITE_VALUE_DEFAULT",           {"hipStreamWriteValueDefault",           CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x0
    {"CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER", {"hipStreamWriteValueNoMemoryBarrier",   CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x1

    // Flags for ::cuStreamBatchMemOp (enum CUstreamBatchMemOpType/CUstreamBatchMemOpType_enum)
    {"CU_STREAM_MEM_OP_WAIT_VALUE_32",          {"hipStreamBatchMemOpWaitValue32",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 1
    {"CU_STREAM_MEM_OP_WRITE_VALUE_32",         {"hipStreamBatchMemOpWriteValue32",      CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 2
    {"CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES",    {"hipStreamBatchMemOpFlushRemoteWrites", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 3

    // Error Handling
    {"cuGetErrorName",   {"hipGetErrorName___",   CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED}},    // cudaGetErrorName (hipGetErrorName) has different signature
    {"cuGetErrorString", {"hipGetErrorString___", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED}},    // cudaGetErrorString (hipGetErrorString) has different signature

    // Init
    {"cuInit", {"hipInit", CONV_INIT, API_DRIVER}},

    // Driver
    {"cuDriverGetVersion", {"hipDriverGetVersion", CONV_VERSION, API_DRIVER}},

    // Context Management
    {"cuCtxCreate_v2",              {"hipCtxCreate",                 CONV_CONTEXT, API_DRIVER}},
    {"cuCtxDestroy_v2",             {"hipCtxDestroy",                CONV_CONTEXT, API_DRIVER}},
    {"cuCtxGetApiVersion",          {"hipCtxGetApiVersion",          CONV_CONTEXT, API_DRIVER}},
    {"cuCtxGetCacheConfig",         {"hipCtxGetCacheConfig",         CONV_CONTEXT, API_DRIVER}},
    {"cuCtxGetCurrent",             {"hipCtxGetCurrent",             CONV_CONTEXT, API_DRIVER}},
    {"cuCtxGetDevice",              {"hipCtxGetDevice",              CONV_CONTEXT, API_DRIVER}},
    {"cuCtxGetFlags",               {"hipCtxGetFlags",               CONV_CONTEXT, API_DRIVER}},
    {"cuCtxGetLimit",               {"hipCtxGetLimit",               CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuCtxGetSharedMemConfig",     {"hipCtxGetSharedMemConfig",     CONV_CONTEXT, API_DRIVER}},
    {"cuCtxGetStreamPriorityRange", {"hipCtxGetStreamPriorityRange", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuCtxPopCurrent_v2",          {"hipCtxPopCurrent",             CONV_CONTEXT, API_DRIVER}},
    {"cuCtxPushCurrent_v2",         {"hipCtxPushCurrent",            CONV_CONTEXT, API_DRIVER}},
    {"cuCtxSetCacheConfig",         {"hipCtxSetCacheConfig",         CONV_CONTEXT, API_DRIVER}},
    {"cuCtxSetCurrent",             {"hipCtxSetCurrent",             CONV_CONTEXT, API_DRIVER}},
    {"cuCtxSetLimit",               {"hipCtxSetLimit",               CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuCtxSetSharedMemConfig",     {"hipCtxSetSharedMemConfig",     CONV_CONTEXT, API_DRIVER}},
    {"cuCtxSynchronize",            {"hipCtxSynchronize",            CONV_CONTEXT, API_DRIVER}},
    // Context Management [DEPRECATED]
    {"cuCtxAttach",                 {"hipCtxAttach",                 CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuCtxDetach",                 {"hipCtxDetach",                 CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},

    // Peer Context Memory Access
    {"cuCtxEnablePeerAccess",   {"hipCtxEnablePeerAccess",   CONV_PEER, API_DRIVER}},
    {"cuCtxDisablePeerAccess",  {"hipCtxDisablePeerAccess",  CONV_PEER, API_DRIVER}},
    {"cuDeviceCanAccessPeer",   {"hipDeviceCanAccessPeer",   CONV_PEER, API_DRIVER}},
    {"cuDeviceGetP2PAttribute", {"hipDeviceGetP2PAttribute", CONV_PEER, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaDeviceGetP2PAttribute)

    // Primary Context Management
    {"cuDevicePrimaryCtxGetState", {"hipDevicePrimaryCtxGetState", CONV_CONTEXT, API_DRIVER}},
    {"cuDevicePrimaryCtxRelease",  {"hipDevicePrimaryCtxRelease",  CONV_CONTEXT, API_DRIVER}},
    {"cuDevicePrimaryCtxReset",    {"hipDevicePrimaryCtxReset",    CONV_CONTEXT, API_DRIVER}},
    {"cuDevicePrimaryCtxRetain",   {"hipDevicePrimaryCtxRetain",   CONV_CONTEXT, API_DRIVER}},
    {"cuDevicePrimaryCtxSetFlags", {"hipDevicePrimaryCtxSetFlags", CONV_CONTEXT, API_DRIVER}},

    // Device Management
    {"cuDeviceGet",           {"hipGetDevice",           CONV_DEVICE, API_DRIVER}},
    {"cuDeviceGetName",       {"hipDeviceGetName",       CONV_DEVICE, API_DRIVER}},
    {"cuDeviceGetCount",      {"hipGetDeviceCount",      CONV_DEVICE, API_DRIVER}},
    {"cuDeviceGetAttribute",  {"hipDeviceGetAttribute",  CONV_DEVICE, API_DRIVER}},
    {"cuDeviceGetPCIBusId",   {"hipDeviceGetPCIBusId",   CONV_DEVICE, API_DRIVER}},
    {"cuDeviceGetByPCIBusId", {"hipDeviceGetByPCIBusId", CONV_DEVICE, API_DRIVER}},
    {"cuDeviceTotalMem_v2",   {"hipDeviceTotalMem",      CONV_DEVICE, API_DRIVER}},

    // Device Management [DEPRECATED]
    {"cuDeviceComputeCapability", {"hipDeviceComputeCapability", CONV_DEVICE, API_DRIVER}},
    {"cuDeviceGetProperties",     {"hipGetDeviceProperties",     CONV_DEVICE, API_DRIVER}},

    // Module Management
    {"cuLinkAddData",         {"hipLinkAddData",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuLinkAddFile",         {"hipLinkAddFile",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuLinkComplete",        {"hipLinkComplete",        CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuLinkCreate",          {"hipLinkCreate",          CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuLinkDestroy",         {"hipLinkDestroy",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuModuleGetFunction",   {"hipModuleGetFunction",   CONV_MODULE, API_DRIVER}},
    {"cuModuleGetGlobal_v2",  {"hipModuleGetGlobal",     CONV_MODULE, API_DRIVER}},
    {"cuModuleGetSurfRef",    {"hipModuleGetSurfRef",    CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuModuleGetTexRef",     {"hipModuleGetTexRef",     CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuModuleLoad",          {"hipModuleLoad",          CONV_MODULE, API_DRIVER}},
    {"cuModuleLoadData",      {"hipModuleLoadData",      CONV_MODULE, API_DRIVER}},
    {"cuModuleLoadDataEx",    {"hipModuleLoadDataEx",    CONV_MODULE, API_DRIVER}},
    {"cuModuleLoadFatBinary", {"hipModuleLoadFatBinary", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuModuleUnload",        {"hipModuleUnload",        CONV_MODULE, API_DRIVER}},

    // enum CUdevice_P2PAttribute/CUdevice_P2PAttribute_enum
    {"CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK",        {"hipDeviceP2PAttributePerformanceRank",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaDevP2PAttrPerformanceRank = 0x01)
    {"CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED",        {"hipDeviceP2PAttributeAccessSupported",       CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaDevP2PAttrAccessSupported = 0x02)
    {"CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED", {"hipDeviceP2PAttributeNativeAtomicSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaDevP2PAttrNativeAtomicSupported = 0x03)

    // Events
    {"CU_EVENT_DEFAULT",        {"hipEventDefault",       CONV_EVENT, API_DRIVER}},
    {"CU_EVENT_BLOCKING_SYNC",  {"hipEventBlockingSync",  CONV_EVENT, API_DRIVER}},
    {"CU_EVENT_DISABLE_TIMING", {"hipEventDisableTiming", CONV_EVENT, API_DRIVER}},
    {"CU_EVENT_INTERPROCESS",   {"hipEventInterprocess",  CONV_EVENT, API_DRIVER}},

    // Event functions
    {"cuEventCreate",           {"hipEventCreate",        CONV_EVENT, API_DRIVER}},
    {"cuEventDestroy_v2",       {"hipEventDestroy",       CONV_EVENT, API_DRIVER}},
    {"cuEventElapsedTime",      {"hipEventElapsedTime",   CONV_EVENT, API_DRIVER}},
    {"cuEventQuery",            {"hipEventQuery",         CONV_EVENT, API_DRIVER}},
    {"cuEventRecord",           {"hipEventRecord",        CONV_EVENT, API_DRIVER}},
    {"cuEventSynchronize",      {"hipEventSynchronize",   CONV_EVENT, API_DRIVER}},

    // Execution Control
    {"cuFuncGetAttribute",       {"hipFuncGetAttribute",       CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuFuncSetCacheConfig",     {"hipFuncSetCacheConfig",     CONV_MODULE, API_DRIVER}},
    {"cuFuncSetSharedMemConfig", {"hipFuncSetSharedMemConfig", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuLaunchKernel",           {"hipModuleLaunchKernel",     CONV_MODULE, API_DRIVER}},

    // Execution Control [DEPRECATED]
    {"cuFuncSetBlockShape", {"hipFuncSetBlockShape", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuFuncSetSharedSize", {"hipFuncSetSharedSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuLaunch",            {"hipLaunch",            CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaLaunch)
    {"cuLaunchGrid",        {"hipLaunchGrid",        CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuLaunchGridAsync",   {"hipLaunchGridAsync",   CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuParamSetf",         {"hipParamSetf",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuParamSeti",         {"hipParamSeti",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuParamSetSize",      {"hipParamSetSize",      CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuParamSetSize",      {"hipParamSetSize",      CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuParamSetv",         {"hipParamSetv",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},

    // Occupancy
    {"cuOccupancyMaxActiveBlocksPerMultiprocessor",          {"hipOccupancyMaxActiveBlocksPerMultiprocessor",          CONV_OCCUPANCY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaOccupancyMaxActiveBlocksPerMultiprocessor)
    {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
    {"cuOccupancyMaxPotentialBlockSize",                     {"hipOccupancyMaxPotentialBlockSize",                     CONV_OCCUPANCY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaOccupancyMaxPotentialBlockSize)
    {"cuOccupancyMaxPotentialBlockSizeWithFlags",            {"hipOccupancyMaxPotentialBlockSizeWithFlags",            CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaOccupancyMaxPotentialBlockSizeWithFlags)

    // Streams
    {"cuStreamAddCallback",        {"hipStreamAddCallback",        CONV_STREAM, API_DRIVER}},
    {"cuStreamAttachMemAsync",     {"hipStreamAttachMemAsync",     CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuStreamCreate",             {"hipStreamCreate__",           CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaStreamCreate due to different signatures
    {"cuStreamCreateWithPriority", {"hipStreamCreateWithPriority", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuStreamDestroy_v2",         {"hipStreamDestroy",            CONV_STREAM, API_DRIVER}},
    {"cuStreamGetFlags",           {"hipStreamGetFlags",           CONV_STREAM, API_DRIVER}},
    {"cuStreamGetPriority",        {"hipStreamGetPriority",        CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuStreamQuery",              {"hipStreamQuery",              CONV_STREAM, API_DRIVER}},
    {"cuStreamSynchronize",        {"hipStreamSynchronize",        CONV_STREAM, API_DRIVER}},
    {"cuStreamWaitEvent",          {"hipStreamWaitEvent",          CONV_STREAM, API_DRIVER}},
    {"cuStreamWaitValue32",        {"hipStreamWaitValue32",        CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE
    {"cuStreamWriteValue32",       {"hipStreamWriteValue32",       CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE
    {"cuStreamBatchMemOp",         {"hipStreamBatchMemOp",         CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE

    // Memory management
    {"cuArray3DCreate",           {"hipArray3DCreate",           CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuArray3DGetDescriptor",    {"hipArray3DGetDescriptor",    CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuArrayCreate",             {"hipArrayCreate",             CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuArrayDestroy",            {"hipArrayDestroy",            CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuArrayGetDescriptor",      {"hipArrayGetDescriptor",      CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuIpcCloseMemHandle",       {"hipIpcCloseMemHandle",       CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuIpcGetEventHandle",       {"hipIpcGetEventHandle",       CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuIpcGetMemHandle",         {"hipIpcGetMemHandle",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuIpcOpenEventHandle",      {"hipIpcOpenEventHandle",      CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuIpcOpenMemHandle",        {"hipIpcOpenMemHandle",        CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemAlloc_v2",             {"hipMalloc",                  CONV_MEM, API_DRIVER}},
    {"cuMemAllocHost",            {"hipMemAllocHost",            CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemAllocManaged",         {"hipMemAllocManaged",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemAllocPitch",           {"hipMemAllocPitch__",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemAllocPitch due to different signatures
    {"cuMemcpy",                  {"hipMemcpy__",                CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy due to different signatures
    {"cuMemcpy2D",                {"hipMemcpy2D__",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy2D due to different signatures
    {"cuMemcpy2DAsync",           {"hipMemcpy2DAsync__",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy2DAsync due to different signatures
    {"cuMemcpy2DUnaligned",       {"hipMemcpy2DUnaligned",       CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpy3D",                {"hipMemcpy3D__",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3D due to different signatures
    {"cuMemcpy3DAsync",           {"hipMemcpy3DAsync__",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3DAsync due to different signatures
    {"cuMemcpy3DPeer",            {"hipMemcpy3DPeer__",          CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3DPeer due to different signatures
    {"cuMemcpy3DPeerAsync",       {"hipMemcpy3DPeerAsync__",     CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3DPeerAsync due to different signatures
    {"cuMemcpyAsync",             {"hipMemcpyAsync__",           CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpyAsync due to different signatures
    {"cuMemcpyAtoA",              {"hipMemcpyAtoA",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpyAtoD",              {"hipMemcpyAtoD",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpyAtoH",              {"hipMemcpyAtoH",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpyAtoHAsync",         {"hipMemcpyAtoHAsync",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpyDtoA",              {"hipMemcpyDtoA",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpyDtoD_v2",           {"hipMemcpyDtoD",              CONV_MEM, API_DRIVER}},
    {"cuMemcpyDtoDAsync_v2",      {"hipMemcpyDtoDAsync",         CONV_MEM, API_DRIVER}},
    {"cuMemcpyDtoH_v2",           {"hipMemcpyDtoH",              CONV_MEM, API_DRIVER}},
    {"cuMemcpyDtoHAsync_v2",      {"hipMemcpyDtoHAsync",         CONV_MEM, API_DRIVER}},
    {"cuMemcpyHtoA",              {"hipMemcpyHtoA",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpyHtoAAsync",         {"hipMemcpyHtoAAsync",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemcpyHtoD_v2",           {"hipMemcpyHtoD",              CONV_MEM, API_DRIVER}},
    {"cuMemcpyHtoDAsync_v2",      {"hipMemcpyHtoDAsync",         CONV_MEM, API_DRIVER}},
    {"cuMemcpyPeerAsync",         {"hipMemcpyPeerAsync__",       CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpyPeerAsync due to different signatures
    {"cuMemcpyPeer",              {"hipMemcpyPeer__",            CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpyPeer due to different signatures
    {"cuMemFree_v2",              {"hipFree",                    CONV_MEM, API_DRIVER}},
    {"cuMemFreeHost",             {"hipHostFree",                CONV_MEM, API_DRIVER}},
    {"cuMemGetAddressRange",      {"hipMemGetAddressRange",      CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemGetInfo_v2",           {"hipMemGetInfo",              CONV_MEM, API_DRIVER}},
    {"cuMemHostAlloc",            {"hipHostMalloc",              CONV_MEM, API_DRIVER}},    // API_Runtime ANALOGUE (cudaHostAlloc)
    {"cuMemHostGetDevicePointer", {"hipMemHostGetDevicePointer", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemHostGetFlags",         {"hipMemHostGetFlags",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemHostRegister_v2",      {"hipHostRegister",            CONV_MEM, API_DRIVER}},    // API_Runtime ANALOGUE (cudaHostAlloc)
    {"cuMemHostUnregister",       {"hipHostUnregister",          CONV_MEM, API_DRIVER}},    // API_Runtime ANALOGUE (cudaHostUnregister)
    {"cuMemsetD16_v2",            {"hipMemsetD16",               CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD16Async",          {"hipMemsetD16Async",          CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD2D16_v2",          {"hipMemsetD2D16",             CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD2D16Async",        {"hipMemsetD2D16Async",        CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD2D32_v2",          {"hipMemsetD2D32",             CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD2D32Async",        {"hipMemsetD2D32Async",        CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD2D8_v2",           {"hipMemsetD2D8",              CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD2D8Async",         {"hipMemsetD2D8Async",         CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD32_v2",            {"hipMemset",                  CONV_MEM, API_DRIVER}},    // API_Runtime ANALOGUE (cudaMemset)
    {"cuMemsetD32Async",          {"hipMemsetAsync",             CONV_MEM, API_DRIVER}},    // API_Runtime ANALOGUE (cudaMemsetAsync)
    {"cuMemsetD8_v2",             {"hipMemsetD8",                CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMemsetD8Async",           {"hipMemsetD8Async",           CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMipmappedArrayCreate",    {"hipMipmappedArrayCreate",    CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMipmappedArrayDestroy",   {"hipMipmappedArrayDestroy",   CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuMipmappedArrayGetLevel",  {"hipMipmappedArrayGetLevel",  CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},

    // Unified Addressing
    {"cuMemPrefetchAsync",      {"hipMemPrefetchAsync__",    CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE (cudaMemPrefetchAsync has different signature)
    {"cuMemAdvise",             {"hipMemAdvise",             CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // // API_Runtime ANALOGUE (cudaMemAdvise)
    {"cuMemRangeGetAttribute",  {"hipMemRangeGetAttribute",  CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // // API_Runtime ANALOGUE (cudaMemRangeGetAttribute)
    {"cuMemRangeGetAttributes", {"hipMemRangeGetAttributes", CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},    // // API_Runtime ANALOGUE (cudaMemRangeGetAttributes)
    {"cuPointerGetAttribute",   {"hipPointerGetAttribute",   CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuPointerGetAttributes",  {"hipPointerGetAttributes",  CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuPointerSetAttribute",   {"hipPointerSetAttribute",   CONV_MEM, API_DRIVER, HIP_UNSUPPORTED}},

    // Texture Reference Mngmnt
    // Texture reference filtering modes
    // enum CUfilter_mode/CUfilter_mode_enum
    {"CU_TR_FILTER_MODE_POINT",     {"hipFilterModePoint",           CONV_TEX, API_DRIVER}},    // 0 // API_Runtime ANALOGUE (cudaFilterModePoint = 0)
    {"CU_TR_FILTER_MODE_LINEAR",    {"hipFilterModeLinear",          CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // 1 // API_Runtime ANALOGUE (cudaFilterModeLinear = 1)

    {"cuTexRefGetAddress",          {"hipTexRefGetAddress",          CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetAddressMode",      {"hipTexRefGetAddressMode",      CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetArray",            {"hipTexRefGetArray",            CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetBorderColor",      {"hipTexRefGetBorderColor",      CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE
    {"cuTexRefGetFilterMode",       {"hipTexRefGetFilterMode",       CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetFlags",            {"hipTexRefGetFlags",            CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetFormat",           {"hipTexRefGetFormat",           CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetMaxAnisotropy",    {"hipTexRefGetMaxAnisotropy",    CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetMipmapFilterMode", {"hipTexRefGetMipmapFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetMipmapLevelBias",  {"hipTexRefGetMipmapLevelBias",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetMipmapLevelClamp", {"hipTexRefGetMipmapLevelClamp", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefGetMipmappedArray",   {"hipTexRefGetMipmappedArray",   CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetAddress",          {"hipTexRefSetAddress",          CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetAddress2D",        {"hipTexRefSetAddress2D",        CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetAddressMode",      {"hipTexRefSetAddressMode",      CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetArray",            {"hipTexRefSetArray",            CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetBorderColor",      {"hipTexRefSetBorderColor",      CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE
    {"cuTexRefSetFilterMode",       {"hipTexRefSetFilterMode",       CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetFlags",            {"hipTexRefSetFlags",            CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetFormat",           {"hipTexRefSetFormat",           CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetMaxAnisotropy",    {"hipTexRefSetMaxAnisotropy",    CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetMipmapFilterMode", {"hipTexRefSetMipmapFilterMode", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetMipmapLevelBias",  {"hipTexRefSetMipmapLevelBias",  CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetMipmapLevelClamp", {"hipTexRefSetMipmapLevelClamp", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefSetMipmappedArray",   {"hipTexRefSetMipmappedArray",   CONV_TEX, API_DRIVER, HIP_UNSUPPORTED}},

    // Texture Reference Mngmnt [DEPRECATED]
    {"cuTexRefCreate",                 {"hipTexRefCreate",                 CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexRefDestroy",                {"hipTexRefDestroy",                CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},

    // Surface Reference Mngmnt
    {"cuSurfRefGetArray",              {"hipSurfRefGetArray",              CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED}},
    {"cuSurfRefSetArray",              {"hipSurfRefSetArray",              CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED}},

    // Texture Object Mngmnt
    {"cuTexObjectCreate",              {"hipTexObjectCreate",              CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexObjectDestroy",             {"hipTexObjectDestroy",             CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexObjectGetResourceDesc",     {"hipTexObjectGetResourceDesc",     CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexObjectGetResourceViewDesc", {"hipTexObjectGetResourceViewDesc", CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},
    {"cuTexObjectGetTextureDesc",      {"hipTexObjectGetTextureDesc",      CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},

    // Surface Object Mngmnt
    {"cuSurfObjectCreate",             {"hipSurfObjectCreate",             CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},
    {"cuSurfObjectDestroy",            {"hipSurfObjectDestroy",            CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},
    {"cuSurfObjectGetResourceDesc",    {"hipSurfObjectGetResourceDesc",    CONV_TEX,     API_DRIVER, HIP_UNSUPPORTED}},

    // Graphics Interoperability
    {"cuGraphicsMapResources",                    {"hipGraphicsMapResources",                    CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsMapResources)
    {"cuGraphicsResourceGetMappedMipmappedArray", {"hipGraphicsResourceGetMappedMipmappedArray", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedMipmappedArray)
    {"cuGraphicsResourceGetMappedPointer",        {"hipGraphicsResourceGetMappedPointer",        CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedPointer)
    {"cuGraphicsResourceSetMapFlags",             {"hipGraphicsResourceSetMapFlags",             CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceSetMapFlags)
    {"cuGraphicsSubResourceGetMappedArray",       {"hipGraphicsSubResourceGetMappedArray",       CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsSubResourceGetMappedArray)
    {"cuGraphicsUnmapResources",                  {"hipGraphicsUnmapResources",                  CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsUnmapResources)
    {"cuGraphicsUnregisterResource",              {"hipGraphicsUnregisterResource",              CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsUnregisterResource)

    // Profiler
    {"cuProfilerInitialize", {"hipProfilerInitialize", CONV_OTHER, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaProfilerInitialize)
    {"cuProfilerStart",      {"hipProfilerStart",      CONV_OTHER, API_DRIVER}},    // API_Runtime ANALOGUE (cudaProfilerStart)
    {"cuProfilerStop",       {"hipProfilerStop",       CONV_OTHER, API_DRIVER}},    // API_Runtime ANALOGUE (cudaProfilerStop)

    // OpenGL Interoperability
    // enum CUGLDeviceList/CUGLDeviceList_enum
    {"CU_GL_DEVICE_LIST_ALL",           {"HIP_GL_DEVICE_LIST_ALL",           CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaGLDeviceListAll)
    {"CU_GL_DEVICE_LIST_CURRENT_FRAME", {"HIP_GL_DEVICE_LIST_CURRENT_FRAME", CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaGLDeviceListCurrentFrame)
    {"CU_GL_DEVICE_LIST_NEXT_FRAME",    {"HIP_GL_DEVICE_LIST_NEXT_FRAME",    CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaGLDeviceListNextFrame)

    {"cuGLGetDevices",                  {"hipGLGetDevices",                  CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLGetDevices)
    {"cuGraphicsGLRegisterBuffer",      {"hipGraphicsGLRegisterBuffer",      CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsGLRegisterBuffer)
    {"cuGraphicsGLRegisterImage",       {"hipGraphicsGLRegisterImage",       CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsGLRegisterImage)
    {"cuWGLGetDevice",                  {"hipWGLGetDevice",                  CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaWGLGetDevice)

    // OpenGL Interoperability [DEPRECATED]
    // enum CUGLmap_flags/CUGLmap_flags_enum
    {"CU_GL_MAP_RESOURCE_FLAGS_NONE",          {"HIP_GL_MAP_RESOURCE_FLAGS_NONE",          CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaGLMapFlagsNone)
    {"CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY",     {"HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY",     CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaGLMapFlagsReadOnly)
    {"CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD", {"HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD", CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaGLMapFlagsWriteDiscard)

    {"cuGLCtxCreate",                          {"hipGLCtxCreate",                          CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuGLInit",                               {"hipGLInit",                               CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuGLMapBufferObject",                    {"hipGLMapBufferObject",                    CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaGLMapBufferObject due to different signatures
    {"cuGLMapBufferObjectAsync",               {"hipGLMapBufferObjectAsync",               CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaGLMapBufferObjectAsync due to different signatures
    {"cuGLRegisterBufferObject",               {"hipGLRegisterBufferObject",               CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLRegisterBufferObject)
    {"cuGLSetBufferObjectMapFlags",            {"hipGLSetBufferObjectMapFlags",            CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLSetBufferObjectMapFlags)
    {"cuGLUnmapBufferObject",                  {"hipGLUnmapBufferObject",                  CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLUnmapBufferObject)
    {"cuGLUnmapBufferObjectAsync",             {"hipGLUnmapBufferObjectAsync",             CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLUnmapBufferObjectAsync)
    {"cuGLUnregisterBufferObject",             {"hipGLUnregisterBufferObject",             CONV_GL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLUnregisterBufferObject)

    // Direct3D 9 Interoperability
    // enum CUd3d9DeviceList/CUd3d9DeviceList_enum
    {"CU_D3D9_DEVICE_LIST_ALL",           {"HIP_D3D9_DEVICE_LIST_ALL",           CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaD3D9DeviceListAll)
    {"CU_D3D9_DEVICE_LIST_CURRENT_FRAME", {"HIP_D3D9_DEVICE_LIST_CURRENT_FRAME", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaD3D9DeviceListCurrentFrame)
    {"CU_D3D9_DEVICE_LIST_NEXT_FRAME",    {"HIP_D3D9_DEVICE_LIST_NEXT_FRAME",    CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaD3D9DeviceListNextFrame)

    {"cuD3D9CtxCreate",                   {"hipD3D9CtxCreate",                   CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuD3D9CtxCreateOnDevice",           {"hipD3D9CtxCreateOnDevice",           CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuD3D9GetDevice",                   {"hipD3D9GetDevice",                   CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9GetDevice)
    {"cuD3D9GetDevices",                  {"hipD3D9GetDevices",                  CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9GetDevices)
    {"cuD3D9GetDirect3DDevice",           {"hipD3D9GetDirect3DDevice",           CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9GetDirect3DDevice)
    {"cuGraphicsD3D9RegisterResource",    {"hipGraphicsD3D9RegisterResource",    CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsD3D9RegisterResource)

    // Direct3D 9 Interoperability [DEPRECATED]
    // enum CUd3d9map_flags/CUd3d9map_flags_enum
    {"CU_D3D9_MAPRESOURCE_FLAGS_NONE",         {"HIP_D3D9_MAPRESOURCE_FLAGS_NONE",         CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaD3D9MapFlagsNone)
    {"CU_D3D9_MAPRESOURCE_FLAGS_READONLY",     {"HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",     CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaD3D9MapFlagsReadOnly)
    {"CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD", {"HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaD3D9MapFlagsWriteDiscard)

    // enum CUd3d9register_flags/CUd3d9register_flags_enum
    {"CU_D3D9_REGISTER_FLAGS_NONE",        {"HIP_D3D9_REGISTER_FLAGS_NONE",        CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaD3D9RegisterFlagsNone)
    {"CU_D3D9_REGISTER_FLAGS_ARRAY",       {"HIP_D3D9_REGISTER_FLAGS_ARRAY",       CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaD3D9RegisterFlagsArray)

    {"cuD3D9MapResources",                 {"hipD3D9MapResources",                 CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9MapResources)
    {"cuD3D9RegisterResource",             {"hipD3D9RegisterResource",             CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9RegisterResource)
    {"cuD3D9ResourceGetMappedArray",       {"hipD3D9ResourceGetMappedArray",       CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedArray)
    {"cuD3D9ResourceGetMappedPitch",       {"hipD3D9ResourceGetMappedPitch",       CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedPitch)
    {"cuD3D9ResourceGetMappedPointer",     {"hipD3D9ResourceGetMappedPointer",     CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedPointer)
    {"cuD3D9ResourceGetMappedSize",        {"hipD3D9ResourceGetMappedSize",        CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedSize)
    {"cuD3D9ResourceGetSurfaceDimensions", {"hipD3D9ResourceGetSurfaceDimensions", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetSurfaceDimensions)
    {"cuD3D9ResourceSetMapFlags",          {"hipD3D9ResourceSetMapFlags",          CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceSetMapFlags)
    {"cuD3D9UnmapResources",               {"hipD3D9UnmapResources",               CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9UnmapResources)
    {"cuD3D9UnregisterResource",           {"hipD3D9UnregisterResource",           CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9UnregisterResource)

    // Direct3D 10 Interoperability
    // enum CUd3d10DeviceList/CUd3d10DeviceList_enum
    {"CU_D3D10_DEVICE_LIST_ALL",           {"HIP_D3D10_DEVICE_LIST_ALL",           CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaD3D10DeviceListAll)
    {"CU_D3D10_DEVICE_LIST_CURRENT_FRAME", {"HIP_D3D10_DEVICE_LIST_CURRENT_FRAME", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaD3D10DeviceListCurrentFrame)
    {"CU_D3D10_DEVICE_LIST_NEXT_FRAME",    {"HIP_D3D10_DEVICE_LIST_NEXT_FRAME",    CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaD3D10DeviceListNextFrame)

    {"cuD3D10GetDevice",                   {"hipD3D10GetDevice",                   CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10GetDevice)
    {"cuD3D10GetDevices",                  {"hipD3D10GetDevices",                  CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10GetDevices)
    {"cuGraphicsD3D10RegisterResource",    {"hipGraphicsD3D10RegisterResource",    CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsD3D10RegisterResource)

    // Direct3D 10 Interoperability [DEPRECATED]
    // enum CUd3d10map_flags/CUd3d10map_flags_enum
    {"CU_D3D10_MAPRESOURCE_FLAGS_NONE",         {"HIP_D3D10_MAPRESOURCE_FLAGS_NONE",         CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaD3D10MapFlagsNone)
    {"CU_D3D10_MAPRESOURCE_FLAGS_READONLY",     {"HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",     CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaD3D10MapFlagsReadOnly)
    {"CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD", {"HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaD3D10MapFlagsWriteDiscard)

    // enum CUd3d10register_flags/CUd3d10register_flags_enum
    {"CU_D3D10_REGISTER_FLAGS_NONE",        {"HIP_D3D10_REGISTER_FLAGS_NONE",        CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x00 // API_Runtime ANALOGUE (cudaD3D10RegisterFlagsNone)
    {"CU_D3D10_REGISTER_FLAGS_ARRAY",       {"HIP_D3D10_REGISTER_FLAGS_ARRAY",       CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaD3D10RegisterFlagsArray)

    {"cuD3D10CtxCreate",                    {"hipD3D10CtxCreate",                    CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuD3D10CtxCreateOnDevice",            {"hipD3D10CtxCreateOnDevice",            CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuD3D10GetDirect3DDevice",            {"hipD3D10GetDirect3DDevice",            CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10GetDirect3DDevice)
    {"cuD3D10MapResources",                 {"hipD3D10MapResources",                 CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10MapResources)
    {"cuD3D10RegisterResource",             {"hipD3D10RegisterResource",             CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10RegisterResource)
    {"cuD3D10ResourceGetMappedArray",       {"hipD3D10ResourceGetMappedArray",       CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedArray)
    {"cuD3D10ResourceGetMappedPitch",       {"hipD3D10ResourceGetMappedPitch",       CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedPitch)
    {"cuD3D10ResourceGetMappedPointer",     {"hipD3D10ResourceGetMappedPointer",     CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedPointer)
    {"cuD3D10ResourceGetMappedSize",        {"hipD3D10ResourceGetMappedSize",        CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedSize)
    {"cuD3D10ResourceGetSurfaceDimensions", {"hipD3D10ResourceGetSurfaceDimensions", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetSurfaceDimensions)
    {"cuD310ResourceSetMapFlags",           {"hipD3D10ResourceSetMapFlags",          CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceSetMapFlags)
    {"cuD3D10UnmapResources",               {"hipD3D10UnmapResources",               CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10UnmapResources)
    {"cuD3D10UnregisterResource",           {"hipD3D10UnregisterResource",           CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10UnregisterResource)

    // Direct3D 11 Interoperability
    // enum CUd3d11DeviceList/CUd3d11DeviceList_enum
    {"CU_D3D11_DEVICE_LIST_ALL",           {"HIP_D3D11_DEVICE_LIST_ALL",           CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // 0x01 // API_Runtime ANALOGUE (cudaD3D11DeviceListAll)
    {"CU_D3D11_DEVICE_LIST_CURRENT_FRAME", {"HIP_D3D11_DEVICE_LIST_CURRENT_FRAME", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // 0x02 // API_Runtime ANALOGUE (cudaD3D11DeviceListCurrentFrame)
    {"CU_D3D11_DEVICE_LIST_NEXT_FRAME",    {"HIP_D3D11_DEVICE_LIST_NEXT_FRAME",    CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // 0x03 // API_Runtime ANALOGUE (cudaD3D11DeviceListNextFrame)

    {"cuD3D11GetDevice",                   {"hipD3D11GetDevice",                   CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D11GetDevice)
    {"cuD3D11GetDevices",                  {"hipD3D11GetDevices",                  CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D11GetDevices)
    {"cuGraphicsD3D11RegisterResource",    {"hipGraphicsD3D11RegisterResource",    CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsD3D11RegisterResource)

    // Direct3D 11 Interoperability [DEPRECATED]
    {"cuD3D11CtxCreate",         {"hipD3D11CtxCreate",         CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuD3D11CtxCreateOnDevice", {"hipD3D11CtxCreateOnDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuD3D11GetDirect3DDevice", {"hipD3D11GetDirect3DDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D11GetDirect3DDevice)

    // VDPAU Interoperability
    {"cuGraphicsVDPAURegisterOutputSurface", {"hipGraphicsVDPAURegisterOutputSurface", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsVDPAURegisterOutputSurface)
    {"cuGraphicsVDPAURegisterVideoSurface",  {"hipGraphicsVDPAURegisterVideoSurface",  CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsVDPAURegisterVideoSurface)
    {"cuVDPAUGetDevice",                     {"hipVDPAUGetDevice",                     CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaVDPAUGetDevice)
    {"cuVDPAUCtxCreate",                     {"hipVDPAUCtxCreate",                     CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE

    // EGL Interoperability
    {"cuEGLStreamConsumerAcquireFrame",     {"hipEGLStreamConsumerAcquireFrame",     CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerAcquireFrame)
    {"cuEGLStreamConsumerConnect",          {"hipEGLStreamConsumerConnect",          CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerConnect)
    {"cuEGLStreamConsumerConnectWithFlags", {"hipEGLStreamConsumerConnectWithFlags", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerConnectWithFlags)
    {"cuEGLStreamConsumerDisconnect",       {"hipEGLStreamConsumerDisconnect",       CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
    {"cuEGLStreamConsumerReleaseFrame",     {"hipEGLStreamConsumerReleaseFrame",     CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerReleaseFrame)
    {"cuEGLStreamProducerConnect",          {"hipEGLStreamProducerConnect",          CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerConnect)
    {"cuEGLStreamProducerDisconnect",       {"hipEGLStreamProducerDisconnect",       CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerDisconnect)
    {"cuEGLStreamProducerPresentFrame",     {"hipEGLStreamProducerPresentFrame",     CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerPresentFrame)
    {"cuEGLStreamProducerReturnFrame",      {"hipEGLStreamProducerReturnFrame",      CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerReturnFrame)
    {"cuGraphicsEGLRegisterImage",          {"hipGraphicsEGLRegisterImage",          CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsEGLRegisterImage)
    {"cuGraphicsResourceGetMappedEglFrame", {"hipGraphicsResourceGetMappedEglFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedEglFrame)

/////////////////////////////// CUDA RT API ///////////////////////////////
    // Data types
    {"cudaDataType_t", {"hipDataType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaDataType",   {"hipDataType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_R_16F",     {"hipR16F",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_C_16F",     {"hipC16F",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_R_32F",     {"hipR32F",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_C_32F",     {"hipC32F",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_R_64F",     {"hipR64F",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_C_64F",     {"hipC64F",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_R_8I",      {"hipR8I",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_C_8I",      {"hipC8I",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_R_8U",      {"hipR8U",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_C_8U",      {"hipC8U",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_R_32I",     {"hipR32I",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_C_32I",     {"hipC32I",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_R_32U",     {"hipR32U",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"CUDA_C_32U",     {"hipC32U",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

    // Library property types
    // IMPORTANT: no cuda prefix
    {"MAJOR_VERSION",         {"hipLibraryMajorVersion",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"MINOR_VERSION",         {"hipLibraryMinorVersion",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"PATCH_LEVEL",           {"hipLibraryPatchVersion",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

    // defines
    {"cudaMemAttachGlobal",                 {"hipMemAttachGlobal",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_MEM_ATTACH_GLOBAL = 0x1)
    {"cudaMemAttachHost",                   {"hipMemAttachHost",                   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_Driver ANALOGUE (CU_MEM_ATTACH_HOST = 0x2)
    {"cudaMemAttachSingle",                 {"hipMemAttachSingle",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x04 // API_Driver ANALOGUE (CU_MEM_ATTACH_SINGLE = 0x4)

    {"cudaOccupancyDefault",                {"hipOccupancyDefault",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x00 // API_Driver ANALOGUE (CU_OCCUPANCY_DEFAULT = 0x0)
    {"cudaOccupancyDisableCachingOverride", {"hipOccupancyDisableCachingOverride", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1)

    // Error API
    {"cudaGetLastError",    {"hipGetLastError",    CONV_ERROR, API_RUNTIME}},
    {"cudaPeekAtLastError", {"hipPeekAtLastError", CONV_ERROR, API_RUNTIME}},
    {"cudaGetErrorName",    {"hipGetErrorName",    CONV_ERROR, API_RUNTIME}},
    {"cudaGetErrorString",  {"hipGetErrorString",  CONV_ERROR, API_RUNTIME}},

    // memcpy
    // memcpy structs
    {"cudaMemcpy3DParms",     {"hipMemcpy3DParms",     CONV_MEM, API_RUNTIME}},
    {"cudaMemcpy3DPeerParms", {"hipMemcpy3DPeerParms", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},

    // memcpy functions
    {"cudaMemcpy",                 {"hipMemcpy",                 CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyToArray",          {"hipMemcpyToArray",          CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyToSymbol",         {"hipMemcpyToSymbol",         CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyToSymbolAsync",    {"hipMemcpyToSymbolAsync",    CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyAsync",            {"hipMemcpyAsync",            CONV_MEM, API_RUNTIME}},
    {"cudaMemcpy2D",               {"hipMemcpy2D",               CONV_MEM, API_RUNTIME}},
    {"cudaMemcpy2DAsync",          {"hipMemcpy2DAsync",          CONV_MEM, API_RUNTIME}},
    {"cudaMemcpy2DToArray",        {"hipMemcpy2DToArray",        CONV_MEM, API_RUNTIME}},
    {"cudaMemcpy2DArrayToArray",   {"hipMemcpy2DArrayToArray",   CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpy2DFromArray",      {"hipMemcpy2DFromArray",      CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpy2DFromArrayAsync", {"hipMemcpy2DFromArrayAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpy2DToArrayAsync",   {"hipMemcpy2DToArrayAsync",   CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpy3D",               {"hipMemcpy3D",               CONV_MEM, API_RUNTIME}},
    {"cudaMemcpy3DAsync",          {"hipMemcpy3DAsync",          CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpy3DPeer",           {"hipMemcpy3DPeer",           CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpy3DPeerAsync",      {"hipMemcpy3DPeerAsync",      CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpyArrayToArray",     {"hipMemcpyArrayToArray",     CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpyFromArrayAsync",   {"hipMemcpyFromArrayAsync",   CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemcpyFromSymbol",       {"hipMemcpyFromSymbol",       CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyFromSymbolAsync",  {"hipMemcpyFromSymbolAsync",  CONV_MEM, API_RUNTIME}},
    {"cudaMemAdvise",              {"hipMemAdvise",              CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},    //
    {"cudaMemRangeGetAttribute",   {"hipMemRangeGetAttribute",   CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},    //
    {"cudaMemRangeGetAttributes",  {"hipMemRangeGetAttributes",  CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},    //

    // Memory advise values
    {"cudaMemAdviseSetReadMostly",                {"hipMemAdviseSetReadMostly",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_READ_MOSTLY = 1)
    {"cudaMemAdviseUnsetReadMostly",              {"hipMemAdviseUnsetReadMostly",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2)
    {"cudaMemAdviseSetPreferredLocation",         {"hipMemAdviseSetPreferredLocation",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3)
    {"cudaMemAdviseUnsetPreferredLocation",       {"hipMemAdviseUnsetPreferredLocation",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 4 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4)
    {"cudaMemAdviseSetAccessedBy",                {"hipMemAdviseSetAccessedBy",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 5 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_ACCESSED_BY = 5)
    {"cudaMemAdviseUnsetAccessedBy",              {"hipMemAdviseUnsetAccessedBy",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 6 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6)

    // CUmem_range_attribute
    {"cudaMemRangeAttributeReadMostly",           {"hipMemRangeAttributeReadMostly",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1)
    {"cudaMemRangeAttributePreferredLocation",    {"hipMemRangeAttributePreferredLocation",    CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2)
    {"cudaMemRangeAttributeAccessedBy",           {"hipMemRangeAttributeAccessedBy",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3)
    {"cudaMemRangeAttributeLastPrefetchLocation", {"hipMemRangeAttributeLastPrefetchLocation", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 4 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4)

    // memcpy kind
    {"cudaMemcpyHostToHost",     {"hipMemcpyHostToHost",     CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyHostToDevice",   {"hipMemcpyHostToDevice",   CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyDeviceToHost",   {"hipMemcpyDeviceToHost",   CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyDeviceToDevice", {"hipMemcpyDeviceToDevice", CONV_MEM, API_RUNTIME}},
    {"cudaMemcpyDefault",        {"hipMemcpyDefault",        CONV_MEM, API_RUNTIME}},

    // memset
    {"cudaMemset",        {"hipMemset",        CONV_MEM, API_RUNTIME}},
    {"cudaMemsetAsync",   {"hipMemsetAsync",   CONV_MEM, API_RUNTIME}},
    {"cudaMemset2D",      {"hipMemset2D",      CONV_MEM, API_RUNTIME}},
    {"cudaMemset2DAsync", {"hipMemset2DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemset3D",      {"hipMemset3D",      CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemset3DAsync", {"hipMemset3DAsync", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},

    // Memory management
    {"cudaMemGetInfo",             {"hipMemGetInfo",             CONV_MEM, API_RUNTIME}},
    {"cudaArrayGetInfo",           {"hipArrayGetInfo",           CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaFreeMipmappedArray",     {"hipFreeMipmappedArray",     CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGetMipmappedArrayLevel", {"hipGetMipmappedArrayLevel", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGetSymbolAddress",       {"hipGetSymbolAddress",       CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGetSymbolSize",          {"hipGetSymbolSize",          CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMemPrefetchAsync",       {"hipMemPrefetchAsync",       CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},    // // API_Driver ANALOGUE (cuMemPrefetchAsync)

    // malloc
    {"cudaMalloc",               {"hipMalloc",               CONV_MEM, API_RUNTIME}},
    {"cudaMallocHost",           {"hipHostMalloc",           CONV_MEM, API_RUNTIME}},
    {"cudaMallocArray",          {"hipMallocArray",          CONV_MEM, API_RUNTIME}},
    {"cudaMalloc3D",             {"hipMalloc3D",             CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMalloc3DArray",        {"hipMalloc3DArray",        CONV_MEM, API_RUNTIME}},
    {"cudaMallocManaged",        {"hipMallocManaged",        CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMallocMipmappedArray", {"hipMallocMipmappedArray", CONV_MEM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaMallocPitch",          {"hipMallocPitch",          CONV_MEM, API_RUNTIME}},

    {"cudaFree",           {"hipFree",           CONV_MEM, API_RUNTIME}},
    {"cudaFreeHost",       {"hipHostFree",       CONV_MEM, API_RUNTIME}},
    {"cudaFreeArray",      {"hipFreeArray",      CONV_MEM, API_RUNTIME}},
    {"cudaHostRegister",   {"hipHostRegister",   CONV_MEM, API_RUNTIME}},
    {"cudaHostUnregister", {"hipHostUnregister", CONV_MEM, API_RUNTIME}},
    // hipHostAlloc deprecated - use hipHostMalloc instead
    {"cudaHostAlloc",      {"hipHostMalloc",     CONV_MEM, API_RUNTIME}},

    // enum cudaMemoryType
    {"cudaMemoryTypeHost",   {"hipMemoryTypeHost",   CONV_MEM, API_RUNTIME}},
    {"cudaMemoryTypeDevice", {"hipMemoryTypeDevice", CONV_MEM, API_RUNTIME}},

    // make memory functions
    {"make_cudaExtent",     {"make_hipExtent",     CONV_MEM, API_RUNTIME}},
    {"make_cudaPitchedPtr", {"make_hipPitchedPtr", CONV_MEM, API_RUNTIME}},
    {"make_cudaPos",        {"make_hipPos",        CONV_MEM, API_RUNTIME}},

    // Host Malloc Flags (#defines)
    {"cudaHostAllocDefault",       {"hipHostMallocDefault",       CONV_MEM, API_RUNTIME}},
    {"cudaHostAllocPortable",      {"hipHostMallocPortable",      CONV_MEM, API_RUNTIME}},
    {"cudaHostAllocMapped",        {"hipHostMallocMapped",        CONV_MEM, API_RUNTIME}},
    {"cudaHostAllocWriteCombined", {"hipHostMallocWriteCombined", CONV_MEM, API_RUNTIME}},

    // Host Register Flags
    {"cudaHostGetFlags",         {"hipHostGetFlags",         CONV_MEM, API_RUNTIME}},
    {"cudaHostRegisterDefault",  {"hipHostRegisterDefault",  CONV_MEM, API_RUNTIME}},
    {"cudaHostRegisterPortable", {"hipHostRegisterPortable", CONV_MEM, API_RUNTIME}},
    {"cudaHostRegisterMapped",   {"hipHostRegisterMapped",   CONV_MEM, API_RUNTIME}},
    {"cudaHostRegisterIoMemory", {"hipHostRegisterIoMemory", CONV_MEM, API_RUNTIME}},

    {"warpSize",    {"hipWarpSize",    CONV_SPECIAL_FUNC, API_RUNTIME}},

    // Events
    {"cudaEventCreate",              {"hipEventCreate",              CONV_EVENT,  API_RUNTIME}},
    {"cudaEventCreateWithFlags",     {"hipEventCreateWithFlags",     CONV_EVENT,  API_RUNTIME}},
    {"cudaEventDestroy",             {"hipEventDestroy",             CONV_EVENT,  API_RUNTIME}},
    {"cudaEventRecord",              {"hipEventRecord",              CONV_EVENT,  API_RUNTIME}},
    {"cudaEventElapsedTime",         {"hipEventElapsedTime",         CONV_EVENT,  API_RUNTIME}},
    {"cudaEventSynchronize",         {"hipEventSynchronize",         CONV_EVENT,  API_RUNTIME}},
    {"cudaEventQuery",               {"hipEventQuery",               CONV_EVENT,  API_RUNTIME}},
    // Event Flags
    {"cudaEventDefault",             {"hipEventDefault",             CONV_EVENT,  API_RUNTIME}},
    {"cudaEventBlockingSync",        {"hipEventBlockingSync",        CONV_EVENT,  API_RUNTIME}},
    {"cudaEventDisableTiming",       {"hipEventDisableTiming",       CONV_EVENT,  API_RUNTIME}},
    {"cudaEventInterprocess",        {"hipEventInterprocess",        CONV_EVENT,  API_RUNTIME}},

    // Streams
    {"cudaStreamCreate",             {"hipStreamCreate",             CONV_STREAM, API_RUNTIME}},
    {"cudaStreamCreateWithFlags",    {"hipStreamCreateWithFlags",    CONV_STREAM, API_RUNTIME}},
    {"cudaStreamCreateWithPriority", {"hipStreamCreateWithPriority", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaStreamDestroy",            {"hipStreamDestroy",            CONV_STREAM, API_RUNTIME}},
    {"cudaStreamWaitEvent",          {"hipStreamWaitEvent",          CONV_STREAM, API_RUNTIME}},
    {"cudaStreamSynchronize",        {"hipStreamSynchronize",        CONV_STREAM, API_RUNTIME}},
    {"cudaStreamGetFlags",           {"hipStreamGetFlags",           CONV_STREAM, API_RUNTIME}},
    {"cudaStreamQuery",              {"hipStreamQuery",              CONV_STREAM, API_RUNTIME}},
    {"cudaStreamAddCallback",        {"hipStreamAddCallback",        CONV_STREAM, API_RUNTIME}},
    {"cudaStreamAttachMemAsync",     {"hipStreamAttachMemAsync",     CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaStreamGetPriority",        {"hipStreamGetPriority",        CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},

    // Stream Flags (defines)
    {"cudaStreamDefault",     {"hipStreamDefault",     CONV_TYPE, API_RUNTIME}},
    {"cudaStreamNonBlocking", {"hipStreamNonBlocking", CONV_TYPE, API_RUNTIME}},

    // Other synchronization
    {"cudaDeviceSynchronize", {"hipDeviceSynchronize", CONV_DEVICE, API_RUNTIME}},
    {"cudaDeviceReset",       {"hipDeviceReset",       CONV_DEVICE, API_RUNTIME}},
    {"cudaSetDevice",         {"hipSetDevice",         CONV_DEVICE, API_RUNTIME}},
    {"cudaGetDevice",         {"hipGetDevice",         CONV_DEVICE, API_RUNTIME}},
    {"cudaGetDeviceCount",    {"hipGetDeviceCount",    CONV_DEVICE, API_RUNTIME}},
    {"cudaChooseDevice",      {"hipChooseDevice",      CONV_DEVICE, API_RUNTIME}},

    // Thread Management
    {"cudaThreadExit",           {"hipDeviceReset",          CONV_THREAD, API_RUNTIME}},
    {"cudaThreadGetCacheConfig", {"hipDeviceGetCacheConfig", CONV_THREAD, API_RUNTIME}},
    {"cudaThreadGetLimit",       {"hipThreadGetLimit",       CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaThreadSetCacheConfig", {"hipDeviceSetCacheConfig", CONV_THREAD, API_RUNTIME}},
    {"cudaThreadSetLimit",       {"hipThreadSetLimit",       CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaThreadSynchronize",    {"hipDeviceSynchronize",    CONV_THREAD, API_RUNTIME}},

    // Attributes
    {"cudaDeviceGetAttribute",                      {"hipDeviceGetAttribute",                              CONV_DEVICE, API_RUNTIME}},

    // enum cudaDeviceAttr
    {"cudaDevAttrMaxThreadsPerBlock",               {"hipDeviceAttributeMaxThreadsPerBlock",               CONV_TYPE,   API_RUNTIME}},                     //  1 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1)
    {"cudaDevAttrMaxBlockDimX",                     {"hipDeviceAttributeMaxBlockDimX",                     CONV_TYPE,   API_RUNTIME}},                     //  2 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2)
    {"cudaDevAttrMaxBlockDimY",                     {"hipDeviceAttributeMaxBlockDimY",                     CONV_TYPE,   API_RUNTIME}},                     //  3 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3)
    {"cudaDevAttrMaxBlockDimZ",                     {"hipDeviceAttributeMaxBlockDimZ",                     CONV_TYPE,   API_RUNTIME}},                     //  4 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4)
    {"cudaDevAttrMaxGridDimX",                      {"hipDeviceAttributeMaxGridDimX",                      CONV_TYPE,   API_RUNTIME}},                     //  5 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5)
    {"cudaDevAttrMaxGridDimY",                      {"hipDeviceAttributeMaxGridDimY",                      CONV_TYPE,   API_RUNTIME}},                     //  6 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 6)
    {"cudaDevAttrMaxGridDimZ",                      {"hipDeviceAttributeMaxGridDimZ",                      CONV_TYPE,   API_RUNTIME}},                     //  7 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 7)
    {"cudaDevAttrMaxSharedMemoryPerBlock",          {"hipDeviceAttributeMaxSharedMemoryPerBlock",          CONV_TYPE,   API_RUNTIME}},                     //  8 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8)
    {"cudaDevAttrTotalConstantMemory",              {"hipDeviceAttributeTotalConstantMemory",              CONV_TYPE,   API_RUNTIME}},                     //  9 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY =9)
    {"cudaDevAttrWarpSize",                         {"hipDeviceAttributeWarpSize",                         CONV_TYPE,   API_RUNTIME}},                     // 10 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10)
    {"cudaDevAttrMaxPitch",                         {"hipDeviceAttributeMaxPitch",                         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 11 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11)
    {"cudaDevAttrMaxRegistersPerBlock",             {"hipDeviceAttributeMaxRegistersPerBlock",             CONV_TYPE,   API_RUNTIME}},                     // 12 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12)
    {"cudaDevAttrClockRate",                        {"hipDeviceAttributeClockRate",                        CONV_TYPE,   API_RUNTIME}},                     // 13 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13)
    {"cudaDevAttrTextureAlignment",                 {"hipDeviceAttributeTextureAlignment",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 14 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14)
    // Is not deprecated as CUDA Driver's API analogue CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
    {"cudaDevAttrGpuOverlap",                       {"hipDeviceAttributeGpuOverlap",                       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 15 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15)
    {"cudaDevAttrMultiProcessorCount",              {"hipDeviceAttributeMultiprocessorCount",              CONV_TYPE,   API_RUNTIME}},                     // 16 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16)
    {"cudaDevAttrKernelExecTimeout",                {"hipDeviceAttributeKernelExecTimeout",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 17 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17)
    {"cudaDevAttrIntegrated",                       {"hipDeviceAttributeIntegrated",                       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 18 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_INTEGRATED = 18)
    {"cudaDevAttrCanMapHostMemory",                 {"hipDeviceAttributeCanMapHostMemory",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 19 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19)
    {"cudaDevAttrComputeMode",                      {"hipDeviceAttributeComputeMode",                      CONV_TYPE,   API_RUNTIME}},                     // 20 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20)
    {"cudaDevAttrMaxTexture1DWidth",                {"hipDeviceAttributeMaxTexture1DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 21 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21)
    {"cudaDevAttrMaxTexture2DWidth",                {"hipDeviceAttributeMaxTexture2DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 22 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22)
    {"cudaDevAttrMaxTexture2DHeight",               {"hipDeviceAttributeMaxTexture2DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 23 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23)
    {"cudaDevAttrMaxTexture3DWidth",                {"hipDeviceAttributeMaxTexture3DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 24 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24)
    {"cudaDevAttrMaxTexture3DHeight",               {"hipDeviceAttributeMaxTexture3DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 25 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25)
    {"cudaDevAttrMaxTexture3DDepth",                {"hipDeviceAttributeMaxTexture3DDepth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 26 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26)
    {"cudaDevAttrMaxTexture2DLayeredWidth",         {"hipDeviceAttributeMaxTexture2DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 27 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27)
    {"cudaDevAttrMaxTexture2DLayeredHeight",        {"hipDeviceAttributeMaxTexture2DLayeredHeight",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 28 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28)
    {"cudaDevAttrMaxTexture2DLayeredLayers",        {"hipDeviceAttributeMaxTexture2DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 29 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29)
    {"cudaDevAttrSurfaceAlignment",                 {"hipDeviceAttributeSurfaceAlignment",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 30 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30)
    {"cudaDevAttrConcurrentKernels",                {"hipDeviceAttributeConcurrentKernels",                CONV_TYPE,   API_RUNTIME}},                     // 31 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31)
    {"cudaDevAttrEccEnabled",                       {"hipDeviceAttributeEccEnabled",                       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 32 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32)
    {"cudaDevAttrPciBusId",                         {"hipDeviceAttributePciBusId",                         CONV_TYPE,   API_RUNTIME}},                     // 33 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33)
    {"cudaDevAttrPciDeviceId",                      {"hipDeviceAttributePciDeviceId",                      CONV_TYPE,   API_RUNTIME}},                     // 34 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34)
    {"cudaDevAttrTccDriver",                        {"hipDeviceAttributeTccDriver",                        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 35 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35)
    {"cudaDevAttrMemoryClockRate",                  {"hipDeviceAttributeMemoryClockRate",                  CONV_TYPE,   API_RUNTIME}},                     // 36 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36)
    {"cudaDevAttrGlobalMemoryBusWidth",             {"hipDeviceAttributeMemoryBusWidth",                   CONV_TYPE,   API_RUNTIME}},                     // 37 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37)
    {"cudaDevAttrL2CacheSize",                      {"hipDeviceAttributeL2CacheSize",                      CONV_TYPE,   API_RUNTIME}},                     // 38 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38)
    {"cudaDevAttrMaxThreadsPerMultiProcessor",      {"hipDeviceAttributeMaxThreadsPerMultiProcessor",      CONV_TYPE,   API_RUNTIME}},                     // 39 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39)
    {"cudaDevAttrAsyncEngineCount",                 {"hipDeviceAttributeAsyncEngineCount",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 40 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40)
    {"cudaDevAttrUnifiedAddressing",                {"hipDeviceAttributeUnifiedAddressing",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 41 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41)
    {"cudaDevAttrMaxTexture1DLayeredWidth",         {"hipDeviceAttributeMaxTexture1DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 42 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42)
    {"cudaDevAttrMaxTexture1DLayeredLayers",        {"hipDeviceAttributeMaxTexture1DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 43 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43)
    // 44 - no
    {"cudaDevAttrMaxTexture2DGatherWidth",          {"hipDeviceAttributeMaxTexture2DGatherWidth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 45 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45)
    {"cudaDevAttrMaxTexture2DGatherHeight",         {"hipDeviceAttributeMaxTexture2DGatherHeight",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 46 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46)
    {"cudaDevAttrMaxTexture3DWidthAlt",             {"hipDeviceAttributeMaxTexture3DWidthAlternate",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 47 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47)
    {"cudaDevAttrMaxTexture3DHeightAlt",            {"hipDeviceAttributeMaxTexture3DHeightAlternate",      CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 48 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48)
    {"cudaDevAttrMaxTexture3DDepthAlt",             {"hipDeviceAttributeMaxTexture3DDepthAlternate",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 49 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49)
    {"cudaDevAttrPciDomainId",                      {"hipDeviceAttributePciDomainId",                      CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 50 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50)
    {"cudaDevAttrTexturePitchAlignment",            {"hipDeviceAttributeTexturePitchAlignment",            CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 51 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51)
    {"cudaDevAttrMaxTextureCubemapWidth",           {"hipDeviceAttributeMaxTextureCubemapWidth",           CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 52 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52)
    {"cudaDevAttrMaxTextureCubemapLayeredWidth",    {"hipDeviceAttributeMaxTextureCubemapLayeredWidth",    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 53 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53)
    {"cudaDevAttrMaxTextureCubemapLayeredLayers",   {"hipDeviceAttributeMaxTextureCubemapLayeredLayers",   CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 54 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54)
    {"cudaDevAttrMaxSurface1DWidth",                {"hipDeviceAttributeMaxSurface1DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 55 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55)
    {"cudaDevAttrMaxSurface2DWidth",                {"hipDeviceAttributeMaxSurface2DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 56 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56)
    {"cudaDevAttrMaxSurface2DHeight",               {"hipDeviceAttributeMaxSurface2DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 57 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57)
    {"cudaDevAttrMaxSurface3DWidth",                {"hipDeviceAttributeMaxSurface3DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 58 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58)
    {"cudaDevAttrMaxSurface3DHeight",               {"hipDeviceAttributeMaxSurface3DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 59 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59)
    {"cudaDevAttrMaxSurface3DDepth",                {"hipDeviceAttributeMaxSurface3DDepth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 60 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60)
    {"cudaDevAttrMaxSurface1DLayeredWidth",         {"hipDeviceAttributeMaxSurface1DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 61 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61)
    {"cudaDevAttrMaxSurface1DLayeredLayers",        {"hipDeviceAttributeMaxSurface1DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 62 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62)
    {"cudaDevAttrMaxSurface2DLayeredWidth",         {"hipDeviceAttributeMaxSurface2DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 63 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63)
    {"cudaDevAttrMaxSurface2DLayeredHeight",        {"hipDeviceAttributeMaxSurface2DLayeredHeight",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 64 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64)
    {"cudaDevAttrMaxSurface2DLayeredLayers",        {"hipDeviceAttributeMaxSurface2DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 65 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65)
    {"cudaDevAttrMaxSurfaceCubemapWidth",           {"hipDeviceAttributeMaxSurfaceCubemapWidth",           CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 66 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66)
    {"cudaDevAttrMaxSurfaceCubemapLayeredWidth",    {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 67 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67)
    {"cudaDevAttrMaxSurfaceCubemapLayeredLayers",   {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",   CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 68 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68)
    {"cudaDevAttrMaxTexture1DLinearWidth",          {"hipDeviceAttributeMaxTexture1DLinearWidth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 69 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69)
    {"cudaDevAttrMaxTexture2DLinearWidth",          {"hipDeviceAttributeMaxTexture2DLinearWidth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 70 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70)
    {"cudaDevAttrMaxTexture2DLinearHeight",         {"hipDeviceAttributeMaxTexture2DLinearHeight",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 71 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71)
    {"cudaDevAttrMaxTexture2DLinearPitch",          {"hipDeviceAttributeMaxTexture2DLinearPitch",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 72 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72)
    {"cudaDevAttrMaxTexture2DMipmappedWidth",       {"hipDeviceAttributeMaxTexture2DMipmappedWidth",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 73 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73)
    {"cudaDevAttrMaxTexture2DMipmappedHeight",      {"hipDeviceAttributeMaxTexture2DMipmappedHeight",      CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 74 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74)
    {"cudaDevAttrComputeCapabilityMajor",           {"hipDeviceAttributeComputeCapabilityMajor",           CONV_TYPE,   API_RUNTIME}},                     // 75 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75)
    {"cudaDevAttrComputeCapabilityMinor",           {"hipDeviceAttributeComputeCapabilityMinor",           CONV_TYPE,   API_RUNTIME}},                     // 76 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76)
    {"cudaDevAttrMaxTexture1DMipmappedWidth",       {"hipDeviceAttributeMaxTexture1DMipmappedWidth",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 77 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77)
    {"cudaDevAttrStreamPrioritiesSupported",        {"hipDeviceAttributeStreamPrioritiesSupported",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 78 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78)
    {"cudaDevAttrGlobalL1CacheSupported",           {"hipDeviceAttributeGlobalL1CacheSupported",           CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 79 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79)
    {"cudaDevAttrLocalL1CacheSupported",            {"hipDeviceAttributeLocalL1CacheSupported",            CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 80 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80)
    {"cudaDevAttrMaxSharedMemoryPerMultiprocessor", {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor", CONV_TYPE,   API_RUNTIME}},                     // 81 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81)
    {"cudaDevAttrMaxRegistersPerMultiprocessor",    {"hipDeviceAttributeMaxRegistersPerMultiprocessor",    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 82 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82)
    {"cudaDevAttrManagedMemory",                    {"hipDeviceAttributeManagedMemory",                    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 83 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83)
    {"cudaDevAttrIsMultiGpuBoard",                  {"hipDeviceAttributeIsMultiGpuBoard",                  CONV_TYPE,   API_RUNTIME}},                     // 84 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84)
    {"cudaDevAttrMultiGpuBoardGroupID",             {"hipDeviceAttributeMultiGpuBoardGroupID",             CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 85 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85)
    {"cudaDevAttrHostNativeAtomicSupported",         {"hipDeviceAttributeHostNativeAtomicSupported",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 86 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86)
    {"cudaDevAttrSingleToDoublePrecisionPerfRatio",  {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio",  CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 87 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87)
    {"cudaDevAttrPageableMemoryAccess",              {"hipDeviceAttributePageableMemoryAccess",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 88 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88)
    {"cudaDevAttrConcurrentManagedAccess",           {"hipDeviceAttributeConcurrentManagedAccess",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 89 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89)
    {"cudaDevAttrComputePreemptionSupported",        {"hipDeviceAttributeComputePreemptionSupported",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 90 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90)
    {"cudaDevAttrCanUseHostPointerForRegisteredMem", {"hipDeviceAttributeCanUseHostPointerForRegisteredMem", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 91 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91)

    // Pointer Attributes
    // struct cudaPointerAttributes
    {"cudaPointerGetAttributes", {"hipPointerGetAttributes", CONV_MEM,  API_RUNTIME}},

    {"cudaHostGetDevicePointer", {"hipHostGetDevicePointer", CONV_MEM,  API_RUNTIME}},

    // Device
    {"cudaGetDeviceProperties",          {"hipGetDeviceProperties",          CONV_DEVICE, API_RUNTIME}},
    {"cudaDeviceGetPCIBusId",            {"hipDeviceGetPCIBusId",            CONV_DEVICE, API_RUNTIME}},
    {"cudaDeviceGetByPCIBusId",          {"hipDeviceGetByPCIBusId",          CONV_DEVICE, API_RUNTIME}},
    {"cudaDeviceGetStreamPriorityRange", {"hipDeviceGetStreamPriorityRange", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaSetValidDevices",              {"hipSetValidDevices",              CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},

    // P2P Attributes (enum cudaDeviceP2PAttr)
    {"cudaDevP2PAttrPerformanceRank",       {"hipDeviceP2PAttributePerformanceRank",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01)
    {"cudaDevP2PAttrAccessSupported",       {"hipDeviceP2PAttributeAccessSupported",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02)
    {"cudaDevP2PAttrNativeAtomicSupported", {"hipDeviceP2PAttributeNativeAtomicSupported", CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x03 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03)
    //
    {"cudaDeviceGetP2PAttribute",           {"hipDeviceGetP2PAttribute",                   CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},    // API_DRIVER ANALOGUE (cuDeviceGetP2PAttribute)

    // enum cudaComputeMode
    {"cudaComputeModeDefault",          {"hipComputeModeDefault",          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_DEFAULT = 0)
    {"cudaComputeModeExclusive",        {"hipComputeModeExclusive",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_EXCLUSIVE = 1)
    {"cudaComputeModeProhibited",       {"hipComputeModeProhibited",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_PROHIBITED = 2)
    {"cudaComputeModeExclusiveProcess", {"hipComputeModeExclusiveProcess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3)

    // Device Flags
    {"cudaGetDeviceFlags", {"hipGetDeviceFlags", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaSetDeviceFlags", {"hipSetDeviceFlags", CONV_DEVICE, API_RUNTIME}},

    // Device stuff (#defines)
    {"cudaDeviceScheduleAuto",         {"hipDeviceScheduleAuto",         CONV_TYPE, API_RUNTIME}},
    {"cudaDeviceScheduleSpin",         {"hipDeviceScheduleSpin",         CONV_TYPE, API_RUNTIME}},
    {"cudaDeviceScheduleYield",        {"hipDeviceScheduleYield",        CONV_TYPE, API_RUNTIME}},
    // deprecated as of CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync
    {"cudaDeviceBlockingSync",         {"hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME}},
    {"cudaDeviceScheduleBlockingSync", {"hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME}},
    {"cudaDeviceScheduleMask",         {"hipDeviceScheduleMask",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaDeviceMapHost",              {"hipDeviceMapHost",              CONV_TYPE, API_RUNTIME}},
    {"cudaDeviceLmemResizeToMax",      {"hipDeviceLmemResizeToMax",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaDeviceMask",                 {"hipDeviceMask",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

    // Cache config
    {"cudaDeviceSetCacheConfig", {"hipDeviceSetCacheConfig", CONV_CACHE, API_RUNTIME}},
    {"cudaDeviceGetCacheConfig", {"hipDeviceGetCacheConfig", CONV_CACHE, API_RUNTIME}},
    {"cudaFuncSetCacheConfig",   {"hipFuncSetCacheConfig",   CONV_CACHE, API_RUNTIME}},

    // Execution control
    // CUDA function cache configurations (enum cudaFuncCache)
    {"cudaFuncCachePreferNone",   {"hipFuncCachePreferNone",   CONV_CACHE, API_RUNTIME}},    // 0 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_NONE = 0x00)
    {"cudaFuncCachePreferShared", {"hipFuncCachePreferShared", CONV_CACHE, API_RUNTIME}},    // 1 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_SHARED = 0x01)
    {"cudaFuncCachePreferL1",     {"hipFuncCachePreferL1",     CONV_CACHE, API_RUNTIME}},    // 2 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_L1 = 0x02)
    {"cudaFuncCachePreferEqual",  {"hipFuncCachePreferEqual",  CONV_CACHE, API_RUNTIME}},    // 3 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_EQUAL = 0x03)

    // Execution control functions
    {"cudaFuncGetAttributes",      {"hipFuncGetAttributes",      CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaFuncSetSharedMemConfig", {"hipFuncSetSharedMemConfig", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGetParameterBuffer",     {"hipGetParameterBuffer",     CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaSetDoubleForDevice",     {"hipSetDoubleForDevice",     CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaSetDoubleForHost",       {"hipSetDoubleForHost",       CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},

    // Execution Control [deprecated since 7.0]
    {"cudaConfigureCall", {"hipConfigureCall", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaLaunch",        {"hipLaunch",        CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaSetupArgument", {"hipSetupArgument", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},

    // Version Management
    {"cudaDriverGetVersion",  {"hipDriverGetVersion",  CONV_VERSION, API_RUNTIME}},
    {"cudaRuntimeGetVersion", {"hipRuntimeGetVersion", CONV_VERSION, API_RUNTIME, HIP_UNSUPPORTED}},

    // Occupancy
    {"cudaOccupancyMaxPotentialBlockSize",                      {"hipOccupancyMaxPotentialBlockSize",                      CONV_OCCUPANCY, API_RUNTIME}},
    {"cudaOccupancyMaxPotentialBlockSizeWithFlags",             {"hipOccupancyMaxPotentialBlockSizeWithFlags",             CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaOccupancyMaxActiveBlocksPerMultiprocessor",           {"hipOccupancyMaxActiveBlocksPerMultiprocessor",           CONV_OCCUPANCY, API_RUNTIME}},
    {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaOccupancyMaxPotentialBlockSizeVariableSMem",          {"hipOccupancyMaxPotentialBlockSizeVariableSMem",          CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", {"hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},

    // Peer2Peer
    {"cudaDeviceCanAccessPeer",     {"hipDeviceCanAccessPeer",     CONV_PEER, API_RUNTIME}},
    {"cudaDeviceDisablePeerAccess", {"hipDeviceDisablePeerAccess", CONV_PEER, API_RUNTIME}},
    {"cudaDeviceEnablePeerAccess",  {"hipDeviceEnablePeerAccess",  CONV_PEER, API_RUNTIME}},

    {"cudaMemcpyPeerAsync",         {"hipMemcpyPeerAsync",         CONV_MEM,  API_RUNTIME}},
    {"cudaMemcpyPeer",              {"hipMemcpyPeer",              CONV_MEM,  API_RUNTIME}},

    // #define cudaIpcMemLazyEnablePeerAccess 0x01
    {"cudaIpcMemLazyEnablePeerAccess", {"hipIpcMemLazyEnablePeerAccess", CONV_TYPE,   API_RUNTIME}},    // 0x01 // API_Driver ANALOGUE (CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1)

    // Shared memory
    {"cudaDeviceSetSharedMemConfig",   {"hipDeviceSetSharedMemConfig",   CONV_DEVICE, API_RUNTIME}},
    {"cudaDeviceGetSharedMemConfig",   {"hipDeviceGetSharedMemConfig",   CONV_DEVICE, API_RUNTIME}},
    // translate deprecated
    //     {"cudaThreadGetSharedMemConfig", {"hipDeviceGetSharedMemConfig", CONV_DEVICE, API_RUNTIME}},
    //     {"cudaThreadSetSharedMemConfig", {"hipDeviceSetSharedMemConfig", CONV_DEVICE, API_RUNTIME}},

    // enum cudaSharedMemConfig
    {"cudaSharedMemBankSizeDefault",   {"hipSharedMemBankSizeDefault",   CONV_TYPE, API_RUNTIME}},
    {"cudaSharedMemBankSizeFourByte",  {"hipSharedMemBankSizeFourByte",  CONV_TYPE, API_RUNTIME}},
    {"cudaSharedMemBankSizeEightByte", {"hipSharedMemBankSizeEightByte", CONV_TYPE, API_RUNTIME}},

    // enum cudaLimit
    {"cudaLimitStackSize",                    {"hipLimitStackSize",                    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x00 // API_Driver ANALOGUE (CU_LIMIT_STACK_SIZE = 0x00)
    {"cudaLimitPrintfFifoSize",               {"hipLimitPrintfFifoSize",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_LIMIT_PRINTF_FIFO_SIZE = 0x01)
    {"cudaLimitMallocHeapSize",               {"hipLimitMallocHeapSize",               CONV_TYPE,   API_RUNTIME}},                     // 0x02 // API_Driver ANALOGUE (CU_LIMIT_MALLOC_HEAP_SIZE = 0x02)
    {"cudaLimitDevRuntimeSyncDepth",          {"hipLimitDevRuntimeSyncDepth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x03 // API_Driver ANALOGUE (CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03)
    {"cudaLimitDevRuntimePendingLaunchCount", {"hipLimitDevRuntimePendingLaunchCount", CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x04 // API_Driver ANALOGUE (CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04)

    {"cudaDeviceGetLimit",                    {"hipDeviceGetLimit",                    CONV_DEVICE, API_RUNTIME}},

    // Profiler
    {"cudaProfilerInitialize", {"hipProfilerInitialize", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuProfilerInitialize)
    {"cudaProfilerStart",      {"hipProfilerStart",      CONV_OTHER, API_RUNTIME}},                     // API_Driver ANALOGUE (cuProfilerStart)
    {"cudaProfilerStop",       {"hipProfilerStop",       CONV_OTHER, API_RUNTIME}},                     // API_Driver ANALOGUE (cuProfilerStop)

    // enum cudaOutputMode
    {"cudaKeyValuePair", {"hipKeyValuePair", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaCSV",          {"hipCSV",          CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED}},

    // Texture Reference Management
    // enum cudaTextureReadMode
    {"cudaReadModeElementType",         {"hipReadModeElementType",         CONV_TEX, API_RUNTIME}},
    {"cudaReadModeNormalizedFloat",     {"hipReadModeNormalizedFloat",     CONV_TEX, API_RUNTIME}},

    // enum cudaTextureFilterMode
    {"cudaFilterModePoint",             {"hipFilterModePoint",             CONV_TEX, API_RUNTIME}},    // 0 // API_DRIVER ANALOGUE (CU_TR_FILTER_MODE_POINT = 0)
    {"cudaFilterModeLinear",            {"hipFilterModeLinear",            CONV_TEX, API_RUNTIME}},    // 1 // API_DRIVER ANALOGUE (CU_TR_FILTER_MODE_POINT = 1)

    {"cudaBindTexture",                 {"hipBindTexture",                 CONV_TEX, API_RUNTIME}},
    {"cudaUnbindTexture",               {"hipUnbindTexture",               CONV_TEX, API_RUNTIME}},
    {"cudaBindTexture2D",               {"hipBindTexture2D",               CONV_TEX, API_RUNTIME}},
    {"cudaBindTextureToArray",          {"hipBindTextureToArray",          CONV_TEX, API_RUNTIME}},
    {"cudaBindTextureToMipmappedArray", {"hipBindTextureToMipmappedArray", CONV_TEX, API_RUNTIME}},    // Unsupported yet on NVCC path
    {"cudaGetTextureAlignmentOffset",   {"hipGetTextureAlignmentOffset",   CONV_TEX, API_RUNTIME}},    // Unsupported yet on NVCC path
    {"cudaGetTextureReference",         {"hipGetTextureReference",         CONV_TEX, API_RUNTIME}},    // Unsupported yet on NVCC path

    // Channel (enum cudaChannelFormatKind)
    {"cudaChannelFormatKindSigned",   {"hipChannelFormatKindSigned",   CONV_TEX, API_RUNTIME}},
    {"cudaChannelFormatKindUnsigned", {"hipChannelFormatKindUnsigned", CONV_TEX, API_RUNTIME}},
    {"cudaChannelFormatKindFloat",    {"hipChannelFormatKindFloat",    CONV_TEX, API_RUNTIME}},
    {"cudaChannelFormatKindNone",     {"hipChannelFormatKindNone",     CONV_TEX, API_RUNTIME}},

    {"cudaCreateChannelDesc",         {"hipCreateChannelDesc",         CONV_TEX, API_RUNTIME}},
    {"cudaGetChannelDesc",            {"hipGetChannelDesc",            CONV_TEX, API_RUNTIME}},

    // Texture Object Management
    // enum cudaResourceType
    {"cudaResourceTypeArray",          {"hipResourceTypeArray",          CONV_TEX, API_RUNTIME}},    // 0x00 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_ARRAY = 0x00)
    {"cudaResourceTypeMipmappedArray", {"hipResourceTypeMipmappedArray", CONV_TEX, API_RUNTIME}},    // 0x01 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01)
    {"cudaResourceTypeLinear",         {"hipResourceTypeLinear",         CONV_TEX, API_RUNTIME}},    // 0x02 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_LINEAR = 0x02)
    {"cudaResourceTypePitch2D",        {"hipResourceTypePitch2D",        CONV_TEX, API_RUNTIME}},    // 0x03 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_PITCH2D = 0x03)

    // enum cudaResourceViewFormat
    {"cudaResViewFormatNone",                       {"hipResViewFormatNone",                       CONV_TEX,      API_RUNTIME}},    // 0x00 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_NONE = 0x00)
    {"cudaResViewFormatUnsignedChar1",              {"hipResViewFormatUnsignedChar1",              CONV_TEX,      API_RUNTIME}},    // 0x01 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01)
    {"cudaResViewFormatUnsignedChar2",              {"hipResViewFormatUnsignedChar2",              CONV_TEX,      API_RUNTIME}},    // 0x02 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02)
    {"cudaResViewFormatUnsignedChar4",              {"hipResViewFormatUnsignedChar4",              CONV_TEX,      API_RUNTIME}},    // 0x03 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03)
    {"cudaResViewFormatSignedChar1",                {"hipResViewFormatSignedChar1",                CONV_TEX,      API_RUNTIME}},    // 0x04 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04)
    {"cudaResViewFormatSignedChar2",                {"hipResViewFormatSignedChar2",                CONV_TEX,      API_RUNTIME}},    // 0x05 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X8 = 0x05)
    {"cudaResViewFormatSignedChar4",                {"hipResViewFormatSignedChar4",                CONV_TEX,      API_RUNTIME}},    // 0x06 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X8 = 0x06)
    {"cudaResViewFormatUnsignedShort1",             {"hipResViewFormatUnsignedShort1",             CONV_TEX,      API_RUNTIME}},    // 0x07 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X16 = 0x07)
    {"cudaResViewFormatUnsignedShort2",             {"hipResViewFormatUnsignedShort2",             CONV_TEX,      API_RUNTIME}},    // 0x08 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X16 = 0x08)
    {"cudaResViewFormatUnsignedShort4",             {"hipResViewFormatUnsignedShort4",             CONV_TEX,      API_RUNTIME}},    // 0x09 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X16 = 0x09)
    {"cudaResViewFormatSignedShort1",               {"hipResViewFormatSignedShort1",               CONV_TEX,      API_RUNTIME}},    // 0x0a // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0a)
    {"cudaResViewFormatSignedShort2",               {"hipResViewFormatSignedShort2",               CONV_TEX,      API_RUNTIME}},    // 0x0b // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0b)
    {"cudaResViewFormatSignedShort4",               {"hipResViewFormatSignedShort4",               CONV_TEX,      API_RUNTIME}},    // 0x0c // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0c)
    {"cudaResViewFormatUnsignedInt1",               {"hipResViewFormatUnsignedInt1",               CONV_TEX,      API_RUNTIME}},    // 0x0d // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0d)
    {"cudaResViewFormatUnsignedInt2",               {"hipResViewFormatUnsignedInt2",               CONV_TEX,      API_RUNTIME}},    // 0x0e // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0e)
    {"cudaResViewFormatUnsignedInt4",               {"hipResViewFormatUnsignedInt4",               CONV_TEX,      API_RUNTIME}},    // 0x0f // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0f)
    {"cudaResViewFormatSignedInt1",                 {"hipResViewFormatSignedInt1",                 CONV_TEX,      API_RUNTIME}},    // 0x10 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X32 = 0x10)
    {"cudaResViewFormatSignedInt2",                 {"hipResViewFormatSignedInt2",                 CONV_TEX,      API_RUNTIME}},    // 0x11 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X32 = 0x11)
    {"cudaResViewFormatSignedInt4",                 {"hipResViewFormatSignedInt4",                 CONV_TEX,      API_RUNTIME}},    // 0x12 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X32 = 0x12)
    {"cudaResViewFormatHalf1",                      {"hipResViewFormatHalf1",                      CONV_TEX,      API_RUNTIME}},    // 0x13 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13)
    {"cudaResViewFormatHalf2",                      {"hipResViewFormatHalf2",                      CONV_TEX,      API_RUNTIME}},    // 0x14 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14)
    {"cudaResViewFormatHalf4",                      {"hipResViewFormatHalf4",                      CONV_TEX,      API_RUNTIME}},    // 0x15 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15)
    {"cudaResViewFormatFloat1",                     {"hipResViewFormatFloat1",                     CONV_TEX,      API_RUNTIME}},    // 0x16 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16)
    {"cudaResViewFormatFloat2",                     {"hipResViewFormatFloat2",                     CONV_TEX,      API_RUNTIME}},    // 0x17 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17)
    {"cudaResViewFormatFloat4",                     {"hipResViewFormatFloat4",                     CONV_TEX,      API_RUNTIME}},    // 0x18 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18)
    {"cudaResViewFormatUnsignedBlockCompressed1",   {"hipResViewFormatUnsignedBlockCompressed1",   CONV_TEX,      API_RUNTIME}},    // 0x19 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19)
    {"cudaResViewFormatUnsignedBlockCompressed2",   {"hipResViewFormatUnsignedBlockCompressed2",   CONV_TEX,      API_RUNTIME}},    // 0x1a // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a)
    {"cudaResViewFormatUnsignedBlockCompressed3",   {"hipResViewFormatUnsignedBlockCompressed3",   CONV_TEX,      API_RUNTIME}},    // 0x1b // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b)
    {"cudaResViewFormatUnsignedBlockCompressed4",   {"hipResViewFormatUnsignedBlockCompressed4",   CONV_TEX,      API_RUNTIME}},    // 0x1c // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c)
    {"cudaResViewFormatSignedBlockCompressed4",     {"hipResViewFormatSignedBlockCompressed4",     CONV_TEX,      API_RUNTIME}},    // 0x1d // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d)
    {"cudaResViewFormatUnsignedBlockCompressed5",   {"hipResViewFormatUnsignedBlockCompressed5",   CONV_TEX,      API_RUNTIME}},    // 0x1e // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e)
    {"cudaResViewFormatSignedBlockCompressed5",     {"hipResViewFormatSignedBlockCompressed5",     CONV_TEX,      API_RUNTIME}},    // 0x1f // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f)
    {"cudaResViewFormatUnsignedBlockCompressed6H",  {"hipResViewFormatUnsignedBlockCompressed6H",  CONV_TEX,      API_RUNTIME}},    // 0x20 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20)
    {"cudaResViewFormatSignedBlockCompressed6H",    {"hipResViewFormatSignedBlockCompressed6H",    CONV_TEX,      API_RUNTIME}},    // 0x21 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21)
    {"cudaResViewFormatUnsignedBlockCompressed7",   {"hipResViewFormatUnsignedBlockCompressed7",   CONV_TEX,      API_RUNTIME}},    // 0x22 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22)

    {"cudaAddressModeWrap",    {"hipAddressModeWrap",    CONV_TEX, API_RUNTIME}},
    {"cudaAddressModeClamp",   {"hipAddressModeClamp",   CONV_TEX, API_RUNTIME}},
    {"cudaAddressModeMirror",  {"hipAddressModeMirror",  CONV_TEX, API_RUNTIME}},
    {"cudaAddressModeBorder",  {"hipAddressModeBorder",  CONV_TEX, API_RUNTIME}},

    // functions
    {"cudaCreateTextureObject",              {"hipCreateTextureObject",              CONV_TEX, API_RUNTIME}},
    {"cudaDestroyTextureObject",             {"hipDestroyTextureObject",             CONV_TEX, API_RUNTIME}},
    {"cudaGetTextureObjectResourceDesc",     {"hipGetTextureObjectResourceDesc",     CONV_TEX, API_RUNTIME}},
    {"cudaGetTextureObjectResourceViewDesc", {"hipGetTextureObjectResourceViewDesc", CONV_TEX, API_RUNTIME}},
    {"cudaGetTextureObjectTextureDesc",      {"hipGetTextureObjectTextureDesc",      CONV_TEX, API_RUNTIME}},

    // Surface Reference Management
    {"cudaBindSurfaceToArray",  {"hipBindSurfaceToArray",  CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGetSurfaceReference", {"hipGetSurfaceReference", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

    // enum cudaSurfaceBoundaryMode
    {"cudaBoundaryModeZero",    {"hipBoundaryModeZero",    CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaBoundaryModeClamp",   {"hipBoundaryModeClamp",   CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaBoundaryModeTrap",    {"hipBoundaryModeTrap",    CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

    // enum cudaSurfaceFormatMode
    {"cudaFormatModeForced",    {"hipFormatModeForced",    CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaFormatModeAuto",      {"hipFormatModeAuto",      CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

    // Surface Object Management
    {"cudaCreateSurfaceObject",          {"hipCreateSurfaceObject",          CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaDestroySurfaceObject",         {"hipDestroySurfaceObject",         CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGetSurfaceObjectResourceDesc", {"hipGetSurfaceObjectResourceDesc", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

    // Inter-Process Communications (IPC)
    {"cudaIpcCloseMemHandle",  {"hipIpcCloseMemHandle",  CONV_DEVICE, API_RUNTIME}},
    {"cudaIpcGetEventHandle",  {"hipIpcGetEventHandle",  CONV_DEVICE, API_RUNTIME}},
    {"cudaIpcGetMemHandle",    {"hipIpcGetMemHandle",    CONV_DEVICE, API_RUNTIME}},
    {"cudaIpcOpenEventHandle", {"hipIpcOpenEventHandle", CONV_DEVICE, API_RUNTIME}},
    {"cudaIpcOpenMemHandle",   {"hipIpcOpenMemHandle",   CONV_DEVICE, API_RUNTIME}},

    // OpenGL Interoperability
    {"cudaGLGetDevices",             {"hipGLGetDevices",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsGLRegisterBuffer", {"hipGraphicsGLRegisterBuffer", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsGLRegisterImage",  {"hipGraphicsGLRegisterImage",  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaWGLGetDevice",             {"hipWGLGetDevice",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},

    // Graphics Interoperability
    {"cudaGraphicsMapResources",                    {"hipGraphicsMapResources",                    CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsMapResources)
    {"cudaGraphicsResourceGetMappedMipmappedArray", {"hipGraphicsResourceGetMappedMipmappedArray", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceGetMappedMipmappedArray)
    {"cudaGraphicsResourceGetMappedPointer",        {"hipGraphicsResourceGetMappedPointer",        CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceGetMappedPointer)
    {"cudaGraphicsResourceSetMapFlags",             {"hipGraphicsResourceSetMapFlags",             CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceSetMapFlags)
    {"cudaGraphicsSubResourceGetMappedArray",       {"hipGraphicsSubResourceGetMappedArray",       CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsSubResourceGetMappedArray)
    {"cudaGraphicsUnmapResources",                  {"hipGraphicsUnmapResources",                  CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsUnmapResources)
    {"cudaGraphicsUnregisterResource",              {"hipGraphicsUnregisterResource",              CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsUnregisterResource)

    // enum cudaGraphicsCubeFace
    {"cudaGraphicsCubeFacePositiveX",               {"hipGraphicsCubeFacePositiveX",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsCubeFaceNegativeX",               {"hipGraphicsCubeFaceNegativeX",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsCubeFacePositiveY",               {"hipGraphicsCubeFacePositiveY",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsCubeFaceNegativeY",               {"hipGraphicsCubeFaceNegativeY",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsCubeFacePositiveZ",               {"hipGraphicsCubeFacePositiveZ",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
    {"cudaGraphicsCubeFaceNegativeZ",               {"hipGraphicsCubeFaceNegativeZ",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},

    // enum cudaGraphicsMapFlags
    {"cudaGraphicsMapFlagsNone",         {"hipGraphicsMapFlagsNone",         CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00)
    {"cudaGraphicsMapFlagsReadOnly",     {"hipGraphicsMapFlagsReadOnly",     CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01)
    {"cudaGraphicsMapFlagsWriteDiscard", {"hipGraphicsMapFlagsWriteDiscard", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02)

    // enum cudaGraphicsRegisterFlags
    {"cudaGraphicsRegisterFlagsNone",             {"hipGraphicsRegisterFlagsNone",             CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00)
    {"cudaGraphicsRegisterFlagsReadOnly",         {"hipGraphicsRegisterFlagsReadOnly",         CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01)
    {"cudaGraphicsRegisterFlagsWriteDiscard",     {"hipGraphicsRegisterFlagsWriteDiscard",     CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02)
    {"cudaGraphicsRegisterFlagsSurfaceLoadStore", {"hipGraphicsRegisterFlagsSurfaceLoadStore", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 4 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04)
    {"cudaGraphicsRegisterFlagsTextureGather",    {"hipGraphicsRegisterFlagsTextureGather",    CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 8 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08)

    // OpenGL Interoperability
    // enum cudaGLDeviceList
    {"cudaGLDeviceListAll",          {"HIP_GL_DEVICE_LIST_ALL",           CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_ALL)
    {"cudaGLDeviceListCurrentFrame", {"HIP_GL_DEVICE_LIST_CURRENT_FRAME", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_CURRENT_FRAME)
    {"cudaGLDeviceListNextFrame",    {"HIP_GL_DEVICE_LIST_NEXT_FRAME",    CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x03 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_NEXT_FRAME)

    {"cudaGLGetDevices",             {"hipGLGetDevices",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLGetDevices)
    {"cudaGraphicsGLRegisterBuffer", {"hipGraphicsGLRegisterBuffer",      CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsGLRegisterBuffer)
    {"cudaGraphicsGLRegisterImage",  {"hipGraphicsGLRegisterImage",       CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsGLRegisterImage)
    {"cudaWGLGetDevice",             {"hipWGLGetDevice",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuWGLGetDevice)

    // OpenGL Interoperability [DEPRECATED]
    // enum cudaGLMapFlags
    {"cudaGLMapFlagsNone",            {"HIP_GL_MAP_RESOURCE_FLAGS_NONE",          CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x00 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_NONE)
    {"cudaGLMapFlagsReadOnly",        {"HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY",     CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY)
    {"cudaGLMapFlagsWriteDiscard",    {"HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD)

    {"cudaGLMapBufferObject",         {"hipGLMapBufferObject__",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // Not equal to cuGLMapBufferObject due to different signatures
    {"cudaGLMapBufferObjectAsync",    {"hipGLMapBufferObjectAsync__",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // Not equal to cuGLMapBufferObjectAsync due to different signatures
    {"cudaGLRegisterBufferObject",    {"hipGLRegisterBufferObject",               CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLRegisterBufferObject)
    {"cudaGLSetBufferObjectMapFlags", {"hipGLSetBufferObjectMapFlags",            CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLSetBufferObjectMapFlags)
    {"cudaGLSetGLDevice",             {"hipGLSetGLDevice",                        CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
    {"cudaGLUnmapBufferObject",       {"hipGLUnmapBufferObject",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnmapBufferObject)
    {"cudaGLUnmapBufferObjectAsync",  {"hipGLUnmapBufferObjectAsync",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnmapBufferObjectAsync)
    {"cudaGLUnregisterBufferObject",  {"hipGLUnregisterBufferObject",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnregisterBufferObject)

    // Direct3D 9 Interoperability
    // enum CUd3d9DeviceList
    {"cudaD3D9DeviceListAll",            {"HIP_D3D9_DEVICE_LIST_ALL",           CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_ALL)
    {"cudaD3D9DeviceListCurrentFrame",   {"HIP_D3D9_DEVICE_LIST_CURRENT_FRAME", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_CURRENT_FRAME)
    {"cudaD3D9DeviceListNextFrame",      {"HIP_D3D9_DEVICE_LIST_NEXT_FRAME",    CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_NEXT_FRAME)

    {"cudaD3D9GetDevice",                {"hipD3D9GetDevice",                   CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDevice)
    {"cudaD3D9GetDevices",               {"hipD3D9GetDevices",                  CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDevices)
    {"cudaD3D9GetDirect3DDevice",        {"hipD3D9GetDirect3DDevice",           CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDirect3DDevice)
    {"cudaD3D9SetDirect3DDevice",        {"hipD3D9SetDirect3DDevice",           CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
    {"cudaGraphicsD3D9RegisterResource", {"hipGraphicsD3D9RegisterResource",    CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D9RegisterResource)

    // Direct3D 9 Interoperability [DEPRECATED]
    // enum cudaD3D9MapFlags
    {"cudaD3D9MapFlags",             {"hipD3D9MapFlags",                         CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d9map_flags)
    {"cudaD3D9MapFlagsNone",         {"HIP_D3D9_MAPRESOURCE_FLAGS_NONE",         CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_NONE)
    {"cudaD3D9MapFlagsReadOnly",     {"HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",     CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_READONLY)
    {"cudaD3D9MapFlagsWriteDiscard", {"HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD)

    // enum cudaD3D9RegisterFlags
    {"cudaD3D9RegisterFlagsNone",            {"HIP_D3D9_REGISTER_FLAGS_NONE",        CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D9_REGISTER_FLAGS_NONE)
    {"cudaD3D9RegisterFlagsArray",           {"HIP_D3D9_REGISTER_FLAGS_ARRAY",       CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D9_REGISTER_FLAGS_ARRAY)

    {"cudaD3D9MapResources",                 {"hipD3D9MapResources",                 CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9MapResources)
    {"cudaD3D9RegisterResource",             {"hipD3D9RegisterResource",             CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9RegisterResource)
    {"cudaD3D9ResourceGetMappedArray",       {"hipD3D9ResourceGetMappedArray",       CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetMappedArray)
    {"cudaD3D9ResourceGetMappedPitch",       {"hipD3D9ResourceGetMappedPitch",       CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cudaD3D9ResourceGetMappedPitch)
    {"cudaD3D9ResourceGetMappedPointer",     {"hipD3D9ResourceGetMappedPointer",     CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetMappedPointer)
    {"cudaD3D9ResourceGetMappedSize",        {"hipD3D9ResourceGetMappedSize",        CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetMappedSize)
    {"cudaD3D9ResourceGetSurfaceDimensions", {"hipD3D9ResourceGetSurfaceDimensions", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetSurfaceDimensions)
    {"cudaD3D9ResourceSetMapFlags",          {"hipD3D9ResourceSetMapFlags",          CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceSetMapFlags)
    {"cudaD3D9UnmapResources",               {"hipD3D9UnmapResources",               CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9UnmapResources)
    {"cudaD3D9UnregisterResource",           {"hipD3D9UnregisterResource",           CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9UnregisterResource)

    // Direct3D 10 Interoperability
    // enum cudaD3D10DeviceList
    {"cudaD3D10DeviceListAll",            {"HIP_D3D10_DEVICE_LIST_ALL",           CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_ALL)
    {"cudaD3D10DeviceListCurrentFrame",   {"HIP_D3D10_DEVICE_LIST_CURRENT_FRAME", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_CURRENT_FRAME)
    {"cudaD3D10DeviceListNextFrame",      {"HIP_D3D10_DEVICE_LIST_NEXT_FRAME",    CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_NEXT_FRAME)
    {"cudaD3D10GetDevice",                {"hipD3D10GetDevice",                   CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10GetDevice)
    {"cudaD3D10GetDevices",               {"hipD3D10GetDevices",                  CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10GetDevices)
    {"cudaGraphicsD3D10RegisterResource", {"hipGraphicsD3D10RegisterResource",    CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D10RegisterResource)

    // Direct3D 10 Interoperability [DEPRECATED]
    // enum cudaD3D10MapFlags
    {"cudaD3D10MapFlagsNone",                 {"HIP_D3D10_MAPRESOURCE_FLAGS_NONE",         CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_NONE)
    {"cudaD3D10MapFlagsReadOnly",             {"HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",     CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_READONLY)
    {"cudaD3D10MapFlagsWriteDiscard",         {"HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD)

    // enum cudaD3D10RegisterFlags
    {"cudaD3D10RegisterFlagsNone",            {"HIP_D3D10_REGISTER_FLAGS_NONE",            CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D10_REGISTER_FLAGS_NONE)
    {"cudaD3D10RegisterFlagsArray",           {"HIP_D3D10_REGISTER_FLAGS_ARRAY",           CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D10_REGISTER_FLAGS_ARRAY)

    {"cudaD3D10GetDirect3DDevice",            {"hipD3D10GetDirect3DDevice",                CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cudaD3D10GetDirect3DDevice)
    {"cudaD3D10MapResources",                 {"hipD3D10MapResources",                     CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10MapResources)
    {"cudaD3D10RegisterResource",             {"hipD3D10RegisterResource",                 CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10RegisterResource)
    {"cudaD3D10ResourceGetMappedArray",       {"hipD3D10ResourceGetMappedArray",           CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetMappedArray)
    {"cudaD3D10ResourceGetMappedPitch",       {"hipD3D10ResourceGetMappedPitch",           CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cudaD3D10ResourceGetMappedPitch)
    {"cudaD3D10ResourceGetMappedPointer",     {"hipD3D10ResourceGetMappedPointer",         CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetMappedPointer)
    {"cudaD3D10ResourceGetMappedSize",        {"hipD3D10ResourceGetMappedSize",            CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetMappedSize)
    {"cudaD3D10ResourceGetSurfaceDimensions", {"hipD3D10ResourceGetSurfaceDimensions",     CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetSurfaceDimensions)
    {"cudaD3D10ResourceSetMapFlags",          {"hipD3D10ResourceSetMapFlags",              CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceSetMapFlags)
    {"cudaD3D10SetDirect3DDevice",            {"hipD3D10SetDirect3DDevice",                CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
    {"cudaD3D10UnmapResources",               {"hipD3D10UnmapResources",                   CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10UnmapResources)
    {"cudaD3D10UnregisterResource",           {"hipD3D10UnregisterResource",               CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10UnregisterResource)

    // Direct3D 11 Interoperability
    // enum cudaD3D11DeviceList
    {"cudaD3D11DeviceListAll",            {"HIP_D3D11_DEVICE_LIST_ALL",           CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_ALL)
    {"cudaD3D11DeviceListCurrentFrame",   {"HIP_D3D11_DEVICE_LIST_CURRENT_FRAME", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_CURRENT_FRAME)
    {"cudaD3D11DeviceListNextFrame",      {"HIP_D3D11_DEVICE_LIST_NEXT_FRAME",    CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_NEXT_FRAME)

    {"cudaD3D11GetDevice",                {"hipD3D11GetDevice",                   CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevice)
    {"cudaD3D11GetDevices",               {"hipD3D11GetDevices",                  CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevices)
    {"cudaGraphicsD3D11RegisterResource", {"hipGraphicsD3D11RegisterResource",    CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D11RegisterResource)

    // Direct3D 11 Interoperability [DEPRECATED]
    {"cudaD3D11GetDevice",                     {"hipD3D11GetDevice",                     CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevice)
    {"cudaD3D11GetDevices",                    {"hipD3D11GetDevices",                    CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevices)
    {"cudaGraphicsD3D11RegisterResource",      {"hipGraphicsD3D11RegisterResource",      CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D11RegisterResource)

    // VDPAU Interoperability
    {"cudaGraphicsVDPAURegisterOutputSurface", {"hipGraphicsVDPAURegisterOutputSurface", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsVDPAURegisterOutputSurface)
    {"cudaGraphicsVDPAURegisterVideoSurface",  {"hipGraphicsVDPAURegisterVideoSurface",  CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsVDPAURegisterVideoSurface)
    {"cudaVDPAUGetDevice",                     {"hipVDPAUGetDevice",                     CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuVDPAUGetDevice)
    {"cudaVDPAUSetVDPAUDevice",                {"hipVDPAUSetDevice",                     CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE

    // EGL Interoperability
    {"cudaEGLStreamConsumerAcquireFrame",     {"hipEGLStreamConsumerAcquireFrame",     CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerAcquireFrame)
    {"cudaEGLStreamConsumerConnect",          {"hipEGLStreamConsumerConnect",          CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerConnect)
    {"cudaEGLStreamConsumerConnectWithFlags", {"hipEGLStreamConsumerConnectWithFlags", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerConnectWithFlags)
    {"cudaEGLStreamConsumerReleaseFrame",     {"hipEGLStreamConsumerReleaseFrame",     CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerReleaseFrame)
    {"cudaEGLStreamProducerConnect",          {"hipEGLStreamProducerConnect",          CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerConnect)
    {"cudaEGLStreamProducerDisconnect",       {"hipEGLStreamProducerDisconnect",       CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerDisconnect)
    {"cudaEGLStreamProducerPresentFrame",     {"hipEGLStreamProducerPresentFrame",     CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerPresentFrame)
    {"cudaEGLStreamProducerReturnFrame",      {"hipEGLStreamProducerReturnFrame",      CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerReturnFrame)
    {"cudaGraphicsEGLRegisterImage",          {"hipGraphicsEGLRegisterImage",          CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsEGLRegisterImage)
    {"cudaGraphicsResourceGetMappedEglFrame", {"hipGraphicsResourceGetMappedEglFrame", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceGetMappedEglFrame)

    ///////////////////////////// cuBLAS /////////////////////////////
    // Blas management functions
    {"cublasInit",                     {"hipblasInit",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasShutdown",                 {"hipblasShutdown",                 CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasGetVersion",               {"hipblasGetVersion",               CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasGetError",                 {"hipblasGetError",                 CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasAlloc",                    {"hipblasAlloc",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasFree",                     {"hipblasFree",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasSetKernelStream",          {"hipblasSetKernelStream",          CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasGetAtomicsMode",           {"hipblasGetAtomicsMode",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasSetAtomicsMode",           {"hipblasSetAtomicsMode",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
 
    // Blas operations (cublasOperation_t)
    {"CUBLAS_OP_N",                    {"HIPBLAS_OP_N",                    CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_OP_T",                    {"HIPBLAS_OP_T",                    CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_OP_C",                    {"HIPBLAS_OP_C",                    CONV_NUMERIC_LITERAL, API_BLAS}},

    // Blas statuses (cublasStatus_t)
    {"CUBLAS_STATUS_SUCCESS",          {"HIPBLAS_STATUS_SUCCESS",          CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_STATUS_NOT_INITIALIZED",  {"HIPBLAS_STATUS_NOT_INITIALIZED",  CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_STATUS_ALLOC_FAILED",     {"HIPBLAS_STATUS_ALLOC_FAILED",     CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_STATUS_INVALID_VALUE",    {"HIPBLAS_STATUS_INVALID_VALUE",    CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_STATUS_MAPPING_ERROR",    {"HIPBLAS_STATUS_MAPPING_ERROR",    CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_STATUS_EXECUTION_FAILED", {"HIPBLAS_STATUS_EXECUTION_FAILED", CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_STATUS_INTERNAL_ERROR",   {"HIPBLAS_STATUS_INTERNAL_ERROR",   CONV_NUMERIC_LITERAL, API_BLAS}},
    {"CUBLAS_STATUS_NOT_SUPPORTED",    {"HIPBLAS_STATUS_INTERNAL_ERROR",   CONV_NUMERIC_LITERAL, API_BLAS}},

    // Blas Fill Modes (cublasFillMode_t)
    {"CUBLAS_FILL_MODE_LOWER",         {"HIPBLAS_FILL_MODE_LOWER",         CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_FILL_MODE_UPPER",         {"HIPBLAS_FILL_MODE_UPPER",         CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

    // Blas Diag Types (cublasDiagType_t)
    {"CUBLAS_DIAG_NON_UNIT",           {"HIPBLAS_DIAG_NON_UNIT",           CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_DIAG_UNIT",               {"HIPBLAS_DIAG_UNIT",               CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

    // Blas Side Modes (cublasSideMode_t
    {"CUBLAS_SIDE_LEFT",               {"HIPBLAS_SIDE_LEFT",               CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_SIDE_RIGHT",              {"HIPBLAS_SIDE_RIGHT",              CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

    // Blas Pointer Modes (cublasPointerMode_t)
    {"CUBLAS_POINTER_MODE_HOST",       {"HIPBLAS_POINTER_MODE_HOST",       CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_POINTER_MODE_DEVICE",     {"HIPBLAS_POINTER_MODE_DEVICE",     CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

    // Blas Atomics Modes (cublasAtomicsMode_t)
    {"CUBLAS_ATOMICS_NOT_ALLOWED",     {"HIPBLAS_ATOMICS_NOT_ALLOWED",     CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_ATOMICS_ALLOWED",         {"HIPBLAS_ATOMICS_ALLOWED",         CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

    // Blas Data Type (cublasDataType_t)
    {"CUBLAS_DATA_FLOAT",              {"HIPBLAS_DATA_FLOAT",              CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_DATA_DOUBLE",             {"HIPBLAS_DATA_DOUBLE",             CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_DATA_HALF",               {"HIPBLAS_DATA_HALF",               CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},
    {"CUBLAS_DATA_INT8",               {"HIPBLAS_DATA_INT8",               CONV_NUMERIC_LITERAL, API_BLAS, HIP_UNSUPPORTED}},

    // Blas1 (v1) Routines
    {"cublasCreate",                   {"hipblasCreate",                   CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDestroy",                  {"hipblasDestroy",                  CONV_MATH_FUNC,       API_BLAS}},

    {"cublasSetVector",                {"hipblasSetVector",                CONV_MATH_FUNC,       API_BLAS}},
    {"cublasGetVector",                {"hipblasGetVector",                CONV_MATH_FUNC,       API_BLAS}},
    {"cublasSetMatrix",                {"hipblasSetMatrix",                CONV_MATH_FUNC,       API_BLAS}},
    {"cublasGetMatrix",                {"hipblasGetMatrix",                CONV_MATH_FUNC,       API_BLAS}},

    {"cublasGetMatrixAsync",           {"hipblasGetMatrixAsync",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasSetMatrixAsync",           {"hipblasSetMatrixAsync",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // NRM2
    {"cublasSnrm2",  {"hipblasSnrm2",  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDnrm2",  {"hipblasDnrm2",  CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasScnrm2", {"hipblasScnrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDznrm2", {"hipblasDznrm2", CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

    // DOT
    {"cublasSdot",        {"hipblasSdot",        CONV_MATH_FUNC, API_BLAS}},
    // there is no such a function in CUDA
    {"cublasSdotBatched", {"hipblasSdotBatched", CONV_MATH_FUNC, API_BLAS}},
    {"cublasDdot",        {"hipblasDdot",        CONV_MATH_FUNC, API_BLAS}},
    // there is no such a function in CUDA
    {"cublasDdotBatched", {"hipblasDdotBatched", CONV_MATH_FUNC, API_BLAS}},
    {"cublasCdotu",       {"hipblasCdotu",       CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCdotc",       {"hipblasCdotc",       CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdotu",       {"hipblasZdotu",       CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdotc",       {"hipblasZdotc",       CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

    // SCAL
    {"cublasSscal",        {"hipblasSscal",        CONV_MATH_FUNC, API_BLAS}},
    // there is no such a function in CUDA
    {"cublasSscalBatched", {"hipblasSscalBatched", CONV_MATH_FUNC, API_BLAS}},
    {"cublasDscal",        {"hipblasDscal",        CONV_MATH_FUNC, API_BLAS}},
    // there is no such a function in CUDA
    {"cublasDscalBatched", {"hipblasDscalBatched", CONV_MATH_FUNC, API_BLAS}},
    {"cublasCscal",        {"hipblasCscal",        CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsscal",       {"hipblasCsscal",       CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZscal",        {"hipblasZscal",        CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdscal",       {"hipblasZdscal",       CONV_MATH_FUNC, API_BLAS, HIP_UNSUPPORTED}},

    // AXPY
    {"cublasSaxpy",                    {"hipblasSaxpy",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasSaxpyBatched",             {"hipblasSaxpyBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDaxpy",                    {"hipblasDaxpy",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasCaxpy",                    {"hipblasCaxpy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZaxpy",                    {"hipblasZaxpy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // COPY
    {"cublasScopy",                    {"hipblasScopy",                    CONV_MATH_FUNC,       API_BLAS}},
    // there is no such a function in CUDA
    {"cublasScopyBatched",             {"hipblasScopyBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDcopy",                    {"hipblasDcopy",                    CONV_MATH_FUNC,       API_BLAS}},
    // there is no such a function in CUDA
    {"cublasDcopyBatched",             {"hipblasDcopyBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasCcopy",                    {"hipblasCcopy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZcopy",                    {"hipblasZcopy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SWAP
    {"cublasSswap",                    {"hipblasSswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDswap",                    {"hipblasDswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCswap",                    {"hipblasCswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZswap",                    {"hipblasZswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // AMAX
    {"cublasIsamax",                   {"hipblasIsamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIdamax",                   {"hipblasIdamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIcamax",                   {"hipblasIcamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIzamax",                   {"hipblasIzamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // AMIN
    {"cublasIsamin",                   {"hipblasIsamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIdamin",                   {"hipblasIdamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIcamin",                   {"hipblasIcamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIzamin",                   {"hipblasIzamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ASUM
    {"cublasSasum",                    {"hipblasSasum",                    CONV_MATH_FUNC,       API_BLAS}},
    // there is no such a function in CUDA
    {"cublasSasumBatched",             {"hipblasSasumBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDasum",                    {"hipblasDasum",                    CONV_MATH_FUNC,       API_BLAS}},
    // there is no such a function in CUDA
    {"cublasDasumBatched",             {"hipblasDasumBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasScasum",                   {"hipblasScasum",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDzasum",                   {"hipblasDzasum",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROT
    {"cublasSrot",                     {"hipblasSrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrot",                     {"hipblasDrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCrot",                     {"hipblasCrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsrot",                    {"hipblasCsrot",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZrot",                     {"hipblasZrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdrot",                    {"hipblasZdrot",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROTG
    {"cublasSrotg",                    {"hipblasSrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrotg",                    {"hipblasDrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCrotg",                    {"hipblasCrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZrotg",                    {"hipblasZrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROTM
    {"cublasSrotm",                    {"hipblasSrotm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrotm",                    {"hipblasDrotm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROTMG
    {"cublasSrotmg",                   {"hipblasSrotmg",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrotmg",                   {"hipblasDrotmg",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // GEMV
    {"cublasSgemv",                    {"hipblasSgemv",                    CONV_MATH_FUNC,       API_BLAS}},
    // there is no such a function in CUDA
    {"cublasSgemvBatched",             {"hipblasSgemvBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDgemv",                    {"hipblasDgemv",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasCgemv",                    {"hipblasCgemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgemv",                    {"hipblasZgemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // GBMV
    {"cublasSgbmv",                    {"hipblasSgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgbmv",                    {"hipblasDgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgbmv",                    {"hipblasCgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgbmv",                    {"hipblasZgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRMV
    {"cublasStrmv",                    {"hipblasStrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrmv",                    {"hipblasDtrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrmv",                    {"hipblasCtrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrmv",                    {"hipblasZtrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TBMV
    {"cublasStbmv",                    {"hipblasStbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtbmv",                    {"hipblasDtbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtbmv",                    {"hipblasCtbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtbmv",                    {"hipblasZtbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TPMV
    {"cublasStpmv",                    {"hipblasStpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtpmv",                    {"hipblasDtpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtpmv",                    {"hipblasCtpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtpmv",                    {"hipblasZtpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRSV
    {"cublasStrsv",                    {"hipblasStrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrsv",                    {"hipblasDtrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrsv",                    {"hipblasCtrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrsv",                    {"hipblasZtrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TPSV
    {"cublasStpsv",                    {"hipblasStpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtpsv",                    {"hipblasDtpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtpsv",                    {"hipblasCtpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtpsv",                    {"hipblasZtpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TBSV
    {"cublasStbsv",                    {"hipblasStbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtbsv",                    {"hipblasDtbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtbsv",                    {"hipblasCtbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtbsv",                    {"hipblasZtbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYMV/HEMV
    {"cublasSsymv",                    {"hipblasSsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsymv",                    {"hipblasDsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsymv",                    {"hipblasCsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsymv",                    {"hipblasZsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChemv",                    {"hipblasChemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhemv",                    {"hipblasZhemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SBMV/HBMV
    {"cublasSsbmv",                    {"hipblasSsbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsbmv",                    {"hpiblasDsbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChbmv",                    {"hipblasChbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhbmv",                    {"hipblasZhbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SPMV/HPMV
    {"cublasSspmv",                    {"hipblasSspmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDspmv",                    {"hipblasDspmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChpmv",                    {"hipblasChpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhpmv",                    {"hipblasZhpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // GER
    {"cublasSger",                     {"hipblasSger",                     CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDger",                     {"hipblasDger",                     CONV_MATH_FUNC,       API_BLAS}},
    {"cublasCgeru",                    {"hipblasCgeru",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgerc",                    {"hipblasCgerc",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgeru",                    {"hipblasZgeru",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgerc",                    {"hipblasZgerc",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYR/HER
    {"cublasSsyr",                     {"hipblasSsyr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyr",                     {"hipblasDsyr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCher",                     {"hipblasCher",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZher",                     {"hipblasZher",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SPR/HPR
    {"cublasSspr",                     {"hipblasSspr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDspr",                     {"hipblasDspr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChpr",                     {"hipblasChpr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhpr",                     {"hipblasZhpr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYR2/HER2
    {"cublasSsyr2",                    {"hipblasSsyr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyr2",                    {"hipblasDsyr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCher2",                    {"hipblasCher2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZher2",                    {"hipblasZher2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SPR2/HPR2
    {"cublasSspr2",                    {"hipblasSspr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDspr2",                    {"hipblasDspr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChpr2",                    {"hipblasChpr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhpr2",                    {"hipblasZhpr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // Blas3 (v1) Routines
    // GEMM
    {"cublasSgemm",                    {"hipblasSgemm",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDgemm",                    {"hipblasDgemm",                    CONV_MATH_FUNC,       API_BLAS}},

    {"cublasCgemm",                    {"hipblasCgemm",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasZgemm",                    {"hipblasZgemm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // BATCH GEMM
    {"cublasSgemmBatched",             {"hipblasSgemmBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDgemmBatched",             {"hipblasDgemmBatched",             CONV_MATH_FUNC,       API_BLAS}},

    {"cublasCgemmBatched",             {"hipblasCgemmBatched",             CONV_MATH_FUNC,       API_BLAS}},
    {"cublasZgemmBatched",             {"hipblasZgemmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYRK
    {"cublasSsyrk",                    {"hipblasSsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyrk",                    {"hipblasDsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsyrk",                    {"hipblasCsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsyrk",                    {"hipblasZsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // HERK
    {"cublasCherk",                    {"hipblasCherk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZherk",                    {"hipblasZherk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYR2K
    {"cublasSsyr2k",                   {"hipblasSsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyr2k",                   {"hipblasDsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsyr2k",                   {"hipblasCsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsyr2k",                   {"hipblasZsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYRKX - eXtended SYRK
    {"cublasSsyrkx",                   {"hipblasSsyrkx",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyrkx",                   {"hipblasDsyrkx",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsyrkx",                   {"hipblasCsyrkx",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsyrkx",                   {"hipblasZsyrkx",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},


    // HER2K
    {"cublasCher2k",                   {"hipblasCher2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZher2k",                   {"hipblasZher2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // HERKX - eXtended HERK
    {"cublasCherkx",                   {"hipblasCherkx",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZherkx",                   {"hipblasZherkx",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYMM
    {"cublasSsymm",                    {"hipblasSsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsymm",                    {"hipblasDsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsymm",                    {"hipblasCsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsymm",                    {"hipblasZsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // HEMM
    {"cublasChemm",                    {"hipblasChemm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhemm",                    {"hipblasZhemm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRSM
    {"cublasStrsm",                    {"hipblasStrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrsm",                    {"hipblasDtrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrsm",                    {"hipblasCtrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrsm",                    {"hipblasZtrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRSM - Batched Triangular Solver
    {"cublasStrsmBatched",             {"hipblasStrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRMM
    {"cublasStrmm",                    {"hipblasStrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrmm",                    {"hipblasDtrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrmm",                    {"hipblasCtrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrmm",                    {"hipblasZtrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ------------------------ CUBLAS BLAS - like extension (cublas_api.h)
    // GEAM
    {"cublasSgeam",                    {"hipblasSgeam",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgeam",                    {"hipblasDgeam",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgeam",                    {"hipblasCgeam",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgeam",                    {"hipblasZgeam",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // GETRF - Batched LU
    {"cublasSgetrfBatched",            {"hipblasSgetrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgetrfBatched",            {"hipblasDgetrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgetrfBatched",            {"hipblasCgetrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgetrfBatched",            {"hipblasZgetrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // Batched inversion based on LU factorization from getrf
    {"cublasSgetriBatched",            {"hipblasSgetriBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgetriBatched",            {"hipblasDgetriBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgetriBatched",            {"hipblasCgetriBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgetriBatched",            {"hipblasZgetriBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // Batched solver based on LU factorization from getrf
    {"cublasSgetrsBatched",            {"hipblasSgetrsBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgetrsBatched",            {"hipblasDgetrsBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgetrsBatched",            {"hipblasCgetrsBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgetrsBatched",            {"hipblasZgetrsBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRSM - Batched Triangular Solver
    {"cublasStrsmBatched",             {"hipblasStrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // MATINV - Batched
    {"cublasSmatinvBatched",           {"hipblasSmatinvBatched",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDmatinvBatched",           {"hipblasDmatinvBatched",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCmatinvBatched",           {"hipblasCmatinvBatched",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZmatinvBatched",           {"hipblasZmatinvBatched",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // Batch QR Factorization
    {"cublasSgeqrfBatched",            {"hipblasSgeqrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgeqrfBatched",            {"hipblasDgeqrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgeqrfBatched",            {"hipblasCgeqrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgeqrfBatched",            {"hipblasZgeqrfBatched",            CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // Least Square Min only m >= n and Non-transpose supported
    {"cublasSgelsBatched",             {"hipblasSgelsBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgelsBatched",             {"hipblasDgelsBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgelsBatched",             {"hipblasCgelsBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgelsBatched",             {"hipblasZgelsBatched",             CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // DGMM
    {"cublasSdgmm",                    {"hipblasSdgmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDdgmm",                    {"hipblasDdgmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCdgmm",                    {"hipblasCdgmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdgmm",                    {"hipblasZdgmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TPTTR - Triangular Pack format to Triangular format
    {"cublasStpttr",                   {"hipblasStpttr",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtpttr",                   {"hipblasDtpttr",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtpttr",                   {"hipblasCtpttr",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtpttr",                   {"hipblasZtpttr",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRTTP - Triangular format to Triangular Pack format
    {"cublasStrttp",                   {"hipblasStrttp",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrttp",                   {"hipblasDtrttp",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrttp",                   {"hipblasCtrttp",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrttp",                   {"hipblasZtrttp",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // Blas2 (v2) Routines
    {"cublasCreate_v2",                {"hipblasCreate",                   CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDestroy_v2",               {"hipblasDestroy",                  CONV_MATH_FUNC,       API_BLAS}},

    {"cublasGetVersion_v2",            {"hipblasGetVersion",               CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasSetStream_v2",             {"hipblasSetStream",                CONV_MATH_FUNC,       API_BLAS}},
    {"cublasGetStream_v2",             {"hipblasGetStream",                CONV_MATH_FUNC,       API_BLAS}},
    {"cublasGetPointerMode_v2",        {"hipblasGetPointerMode",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasSetPointerMode_v2",        {"hipblasSetPointerMode",           CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // GEMV
    {"cublasSgemv_v2",                 {"hipblasSgemv",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDgemv_v2",                 {"hipblasDgemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgemv_v2",                 {"hipblasCgemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgemv_v2",                 {"hipblasZgemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // GBMV
    {"cublasSgbmv_v2",                 {"hipblasSgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDgbmv_v2",                 {"hipblasDgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgbmv_v2",                 {"hipblasCgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgbmv_v2",                 {"hipblasZgbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRMV
    {"cublasStrmv_v2",                 {"hipblasStrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrmv_v2",                 {"hipblasDtrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrmv_v2",                 {"hipblasCtrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrmv_v2",                 {"hipblasZtrmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TBMV
    {"cublasStbmv_v2",                 {"hipblasStbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtbmv_v2",                 {"hipblasDtbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtbmv_v2",                 {"hipblasCtbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtbmv_v2",                 {"hipblasZtbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TPMV
    {"cublasStpmv_v2",                 {"hipblasStpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtpmv_v2",                 {"hipblasDtpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtpmv_v2",                 {"hipblasCtpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtpmv_v2",                 {"hipblasZtpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRSV
    {"cublasStrsv_v2",                 {"hipblasStrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrsv_v2",                 {"hipblasDtrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrsv_v2",                 {"hipblasCtrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrsv_v2",                 {"hipblasZtrsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TPSV
    {"cublasStpsv_v2",                 {"hipblasStpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtpsv_v2",                 {"hipblasDtpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtpsv_v2",                 {"hipblasCtpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtpsv_v2",                 {"hipblasZtpsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TBSV
    {"cublasStbsv_v2",                 {"hipblasStbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtbsv_v2",                 {"hipblasDtbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtbsv_v2",                 {"hipblasCtbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtbsv_v2",                 {"hipblasZtbsv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYMV/HEMV
    {"cublasSsymv_v2",                 {"hipblasSsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsymv_v2",                 {"hipblasDsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsymv_v2",                 {"hipblasCsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsymv_v2",                 {"hipblasZsymv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChemv_v2",                 {"hipblasChemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhemv_v2",                 {"hipblasZhemv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SBMV/HBMV
    {"cublasSsbmv_v2",                 {"hipblasSsbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsbmv_v2",                 {"hpiblasDsbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChbmv_v2",                 {"hipblasChbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhbmv_v2",                 {"hipblasZhbmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SPMV/HPMV
    {"cublasSspmv_v2",                 {"hipblasSspmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDspmv_v2",                 {"hipblasDspmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChpmv_v2",                 {"hipblasChpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhpmv_v2",                 {"hipblasZhpmv",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // GER
    {"cublasSger_v2",                  {"hipblasSger",                     CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDger_v2",                  {"hipblasDger",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgeru_v2",                 {"hipblasCgeru",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCgerc_v2",                 {"hipblasCgerc",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgeru_v2",                 {"hipblasZgeru",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZgerc_v2",                 {"hipblasZgerc",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYR/HER
    {"cublasSsyr_v2",                  {"hipblasSsyr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyr_v2",                  {"hipblasDsyr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsyr_v2",                  {"hipblasCsyr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsyr_v2",                  {"hipblasZsyr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCher_v2",                  {"hipblasCher",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZher_v2",                  {"hipblasZher",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SPR/HPR
    {"cublasSspr_v2",                  {"hipblasSspr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDspr_v2",                  {"hipblasDspr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChpr_v2",                  {"hipblasChpr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhpr_v2",                  {"hipblasZhpr",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYR2/HER2
    {"cublasSsyr2_v2",                 {"hipblasSsyr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyr2_v2",                 {"hipblasDsyr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsyr2_v2",                 {"hipblasCsyr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsyr2_v2",                 {"hipblasZsyr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCher2_v2",                 {"hipblasCher2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZher2_v2",                 {"hipblasZher2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SPR2/HPR2
    {"cublasSspr2_v2",                 {"hipblasSspr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDspr2_v2",                 {"hipblasDspr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasChpr2_v2",                 {"hipblasChpr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhpr2_v2",                 {"hipblasZhpr2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // Blas3 (v2) Routines
    // GEMM
    {"cublasSgemm_v2",                 {"hipblasSgemm",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDgemm_v2",                 {"hipblasDgemm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    {"cublasCgemm_v2",                 {"hipblasCgemm",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasZgemm_v2",                 {"hipblasZgemm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    //IO in FP16 / FP32, computation in float
    {"cublasSgemmEx",                  {"hipblasSgemmEx",                  CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYRK
    {"cublasSsyrk_v2",                 {"hipblasSsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyrk_v2",                 {"hipblasDsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsyrk_v2",                 {"hipblasCsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsyrk_v2",                 {"hipblasZsyrk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // HERK
    {"cublasCherk_v2",                 {"hipblasCherk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZherk_v2",                 {"hipblasZherk",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYR2K
    {"cublasSsyr2k_v2",                {"hipblasSsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsyr2k_v2",                {"hipblasDsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsyr2k_v2",                {"hipblasCsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsyr2k_v2",                {"hipblasZsyr2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // HER2K
    {"cublasCher2k_v2",                {"hipblasCher2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZher2k_v2",                {"hipblasZher2k",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SYMM
    {"cublasSsymm_v2",                 {"hipblasSsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDsymm_v2",                 {"hipblasDsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsymm_v2",                 {"hipblasCsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZsymm_v2",                 {"hipblasZsymm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // HEMM
    {"cublasChemm_v2",                 {"hipblasChemm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZhemm_v2",                 {"hipblasZhemm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRSM
    {"cublasStrsm_v2",                 {"hipblasStrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrsm_v2",                 {"hipblasDtrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrsm_v2",                 {"hipblasCtrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrsm_v2",                 {"hipblasZtrsm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // TRMM
    {"cublasStrmm_v2",                 {"hipblasStrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDtrmm_v2",                 {"hipblasDtrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCtrmm_v2",                 {"hipblasCtrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZtrmm_v2",                 {"hipblasZtrmm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // NRM2
    {"cublasSnrm2_v2",                 {"hipblasSnrm2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDnrm2_v2",                 {"hipblasDnrm2",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasScnrm2_v2",                {"hipblasScnrm2",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDznrm2_v2",                {"hipblasDznrm2",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // DOT
    {"cublasSdot_v2",                  {"hipblasSdot",                     CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDdot_v2",                  {"hipblasDdot",                     CONV_MATH_FUNC,       API_BLAS}},

    {"cublasCdotu_v2",                 {"hipblasCdotu",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCdotc_v2",                 {"hipblasCdotc",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdotu_v2",                 {"hipblasZdotu",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdotc_v2",                 {"hipblasZdotc",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SCAL
    {"cublasSscal_v2",                 {"hipblasSscal",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDscal_v2",                 {"hipblasDscal",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasCscal_v2",                 {"hipblasCscal",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsscal_v2",                {"hipblasCsscal",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZscal_v2",                 {"hipblasZscal",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdscal_v2",                {"hipblasZdscal",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // AXPY
    {"cublasSaxpy_v2",                 {"hipblasSaxpy",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDaxpy_v2",                 {"hipblasDaxpy",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasCaxpy_v2",                 {"hipblasCaxpy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZaxpy_v2",                 {"hipblasZaxpy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // COPY
    {"cublasScopy_v2",                 {"hipblasScopy",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDcopy_v2",                 {"hipblasDcopy",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasCcopy_v2",                 {"hipblasCcopy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZcopy_v2",                 {"hipblasZcopy",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // SWAP
    {"cublasSswap_v2",                 {"hipblasSswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDswap_v2",                 {"hipblasDswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCswap_v2",                 {"hipblasCswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZswap_v2",                 {"hipblasZswap",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // AMAX
    {"cublasIsamax_v2",                {"hipblasIsamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIdamax_v2",                {"hipblasIdamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIcamax_v2",                {"hipblasIcamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIzamax_v2",                {"hipblasIzamax",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // AMIN
    {"cublasIsamin_v2",                {"hipblasIsamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIdamin_v2",                {"hipblasIdamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIcamin_v2",                {"hipblasIcamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasIzamin_v2",                {"hipblasIzamin",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ASUM
    {"cublasSasum_v2",                 {"hipblasSasum",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasDasum_v2",                 {"hipblasDasum",                    CONV_MATH_FUNC,       API_BLAS}},
    {"cublasScasum_v2",                {"hipblasScasum",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDzasum_v2",                {"hipblasDzasum",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROT
    {"cublasSrot_v2",                  {"hipblasSrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrot_v2",                  {"hipblasDrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCrot_v2",                  {"hipblasCrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCsrot_v2",                 {"hipblasCsrot",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZrot_v2",                  {"hipblasZrot",                     CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZdrot_v2",                 {"hipblasZdrot",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROTG
    {"cublasSrotg_v2",                 {"hipblasSrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrotg_v2",                 {"hipblasDrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasCrotg_v2",                 {"hipblasCrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasZrotg_v2",                 {"hipblasZrotg",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROTM
    {"cublasSrotm_v2",                 {"hipblasSrotm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrotm_v2",                 {"hipblasDrotm",                    CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},

    // ROTMG
    {"cublasSrotmg_v2",                {"hipblasSrotmg",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}},
    {"cublasDrotmg_v2",                {"hipblasDrotmg",                   CONV_MATH_FUNC,       API_BLAS, HIP_UNSUPPORTED}}
};

const std::map<llvm::StringRef, hipCounter>& CUDA_RENAMES_MAP() {
    static std::map<llvm::StringRef, hipCounter> ret;
    if (!ret.empty()) {
        return ret;
    }

    // First run, so compute the union map.
    ret = CUDA_IDENTIFIER_MAP;
    ret.insert(CUDA_TYPE_NAME_MAP.begin(), CUDA_TYPE_NAME_MAP.end());

    return ret;
};
