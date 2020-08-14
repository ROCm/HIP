# HIP Deprecated APIs

## HIP Context Management APIs

CUDA supports cuCtx API, the Driver API that defines "Context" and "Devices" as separate entities. Contexts contain a single device, and a device can theoretically have multiple contexts. HIP initially added limited support for these API to facilitate easy porting from existing driver codes. These API are marked as deprecated now since there are better alternate interface (such as hipSetDevice or the stream API) to achieve the required functions.

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

## HIP Memory Management APIs

### hipMallocHost
Should use "hipHostMalloc" instead.

### hipMemAllocHost
Should use "hipHostMalloc" instead.

### hipHostAlloc
Should use "hipHostMalloc" instead.

### hipFreeHost
Should use "hipHostFree" instead.