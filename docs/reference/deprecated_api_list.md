# HIP Deprecated Runtime Functions


## HIP Context Management APIs

CUDA supports cuCtx API, the Driver API that defines "Context" and "Devices" as separate entities. Contexts contain a single device, and a device can theoretically have multiple contexts. HIP initially added limited support for these API to facilitate easy porting from existing driver codes. These API are marked as deprecated now since there are better alternate interface (such as hipSetDevice or the stream API) to achieve the required functions.
### hipCtxCreate
### hipCtxDestroy
### hipCtxPopCurrent
### hipCtxPushCurrent
### hipCtxSetCurrent
### hipCtxGetCurrent
### hipCtxGetDevice
### hipCtxGetApiVersion
### hipCtxGetCacheConfig
### hipCtxSetCacheConfig
### hipCtxSetSharedMemConfig
### hipCtxGetSharedMemConfig
### hipCtxSynchronize
### hipCtxGetFlags
### hipCtxEnablePeerAccess
### hipCtxDisablePeerAccess
### hipDevicePrimaryCtxGetState
### hipDevicePrimaryCtxRelease
### hipDevicePrimaryCtxRetain
### hipDevicePrimaryCtxReset
### hipDevicePrimaryCtxSetFlags


## HIP Memory Management APIs

### hipMallocHost
Should use "hipHostMalloc" instead.

### hipMemAllocHost
Should use "hipHostMalloc" instead.

### hipHostAlloc
Should use "hipHostMalloc" instead.

### hipFreeHost
Should use "hipHostFree" instead.

### hipMemcpyToArray
### hipMemcpyFromArray


## HIP Profiler Control APIs

### hipProfilerStart
Should use roctracer/rocTX instead

### hipProfilerStop
Should use roctracer/rocTX instead


## HIP Texture Management APIs

### hipGetTextureReference
### hipTexRefSetAddressMode
### hipTexRefSetArray
### hipTexRefSetFilterMode
### hipTexRefSetFlags
### hipTexRefSetFormat
### hipBindTexture
### hipBindTexture2D
### hipBindTextureToArray
### hipGetTextureAlignmentOffset
### hipUnbindTexture
### hipTexRefGetAddress
### hipTexRefGetAddressMode
### hipTexRefGetFilterMode
### hipTexRefGetFlags
### hipTexRefGetFormat
### hipTexRefGetMaxAnisotropy
### hipTexRefGetMipmapFilterMode
### hipTexRefGetMipmapLevelBias
### hipTexRefGetMipmapLevelClamp
### hipTexRefGetMipMappedArray
### hipTexRefSetAddress
### hipTexRefSetAddress2D
### hipTexRefSetMaxAnisotropy
### hipTexRefSetBorderColor
### hipTexRefSetMipmapFilterMode
### hipTexRefSetMipmapLevelBias
### hipTexRefSetMipmapLevelClamp
### hipTexRefSetMipmappedArray
### hipBindTextureToMipmappedArray
### hipTexRefGetBorderColor
### hipTexRefGetArray
