#include "CUDA2HIP.h"

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_FUNCTION_MAP{

/////////////////////////////// CUDA RT API ///////////////////////////////

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

  // memset
  {"cudaMemset",        {"hipMemset",        CONV_MEM, API_RUNTIME}},
  {"cudaMemsetAsync",   {"hipMemsetAsync",   CONV_MEM, API_RUNTIME}},
  {"cudaMemset2D",      {"hipMemset2D",      CONV_MEM, API_RUNTIME}},
  {"cudaMemset2DAsync", {"hipMemset2DAsync", CONV_MEM, API_RUNTIME}},
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
  {"cudaMalloc3D",             {"hipMalloc3D",             CONV_MEM, API_RUNTIME}},
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

  // Device Flags
  {"cudaGetDeviceFlags", {"hipGetDeviceFlags", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaSetDeviceFlags", {"hipSetDeviceFlags", CONV_DEVICE, API_RUNTIME}},

  // Cache config
  {"cudaDeviceSetCacheConfig", {"hipDeviceSetCacheConfig", CONV_CACHE, API_RUNTIME}},
  {"cudaDeviceGetCacheConfig", {"hipDeviceGetCacheConfig", CONV_CACHE, API_RUNTIME}},
  {"cudaFuncSetCacheConfig",   {"hipFuncSetCacheConfig",   CONV_CACHE, API_RUNTIME}},


  // Execution control functions
  {"cudaFuncGetAttributes",      {"hipFuncGetAttributes",      CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaFuncSetSharedMemConfig", {"hipFuncSetSharedMemConfig", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGetParameterBuffer",     {"hipGetParameterBuffer",     CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaSetDoubleForDevice",     {"hipSetDoubleForDevice",     CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaSetDoubleForHost",       {"hipSetDoubleForHost",       CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED}},

  // Execution Control [deprecated since 7.0]
  {"cudaConfigureCall", {"hipConfigureCall", CONV_EXEC, API_RUNTIME}},
  {"cudaLaunch",        {"hipLaunchByPtr",   CONV_EXEC, API_RUNTIME}},
  {"cudaSetupArgument", {"hipSetupArgument", CONV_EXEC, API_RUNTIME}},

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

  // Shared memory
  {"cudaDeviceSetSharedMemConfig",   {"hipDeviceSetSharedMemConfig",   CONV_DEVICE, API_RUNTIME}},
  {"cudaDeviceGetSharedMemConfig",   {"hipDeviceGetSharedMemConfig",   CONV_DEVICE, API_RUNTIME}},
  // translate deprecated
  //     {"cudaThreadGetSharedMemConfig", {"hipDeviceGetSharedMemConfig", CONV_DEVICE, API_RUNTIME}},
  //     {"cudaThreadSetSharedMemConfig", {"hipDeviceSetSharedMemConfig", CONV_DEVICE, API_RUNTIME}},


  {"cudaDeviceGetLimit",                    {"hipDeviceGetLimit",                    CONV_DEVICE, API_RUNTIME}},

  // Profiler
  {"cudaProfilerInitialize", {"hipProfilerInitialize", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuProfilerInitialize)
  {"cudaProfilerStart",      {"hipProfilerStart",      CONV_OTHER, API_RUNTIME}},                     // API_Driver ANALOGUE (cuProfilerStart)
  {"cudaProfilerStop",       {"hipProfilerStop",       CONV_OTHER, API_RUNTIME}},                     // API_Driver ANALOGUE (cuProfilerStop)


  {"cudaBindTexture",                 {"hipBindTexture",                 CONV_TEX, API_RUNTIME}},
  {"cudaUnbindTexture",               {"hipUnbindTexture",               CONV_TEX, API_RUNTIME}},
  {"cudaBindTexture2D",               {"hipBindTexture2D",               CONV_TEX, API_RUNTIME}},
  {"cudaBindTextureToArray",          {"hipBindTextureToArray",          CONV_TEX, API_RUNTIME}},
  {"cudaBindTextureToMipmappedArray", {"hipBindTextureToMipmappedArray", CONV_TEX, API_RUNTIME}},    // Unsupported yet on NVCC path
  {"cudaGetTextureAlignmentOffset",   {"hipGetTextureAlignmentOffset",   CONV_TEX, API_RUNTIME}},    // Unsupported yet on NVCC path
  {"cudaGetTextureReference",         {"hipGetTextureReference",         CONV_TEX, API_RUNTIME}},    // Unsupported yet on NVCC path

  {"cudaCreateChannelDesc",         {"hipCreateChannelDesc",         CONV_TEX, API_RUNTIME}},
  {"cudaGetChannelDesc",            {"hipGetChannelDesc",            CONV_TEX, API_RUNTIME}},

  // Texture Object Management

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


  // Surface Object Management
  {"cudaCreateSurfaceObject",          {"hipCreateSurfaceObject",          CONV_SURFACE, API_RUNTIME}},
  {"cudaDestroySurfaceObject",         {"hipDestroySurfaceObject",         CONV_SURFACE, API_RUNTIME}},
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

  {"cudaGLGetDevices",             {"hipGLGetDevices",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLGetDevices)
  {"cudaGraphicsGLRegisterBuffer", {"hipGraphicsGLRegisterBuffer",      CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsGLRegisterBuffer)
  {"cudaGraphicsGLRegisterImage",  {"hipGraphicsGLRegisterImage",       CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsGLRegisterImage)
  {"cudaWGLGetDevice",             {"hipWGLGetDevice",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuWGLGetDevice)

  // OpenGL Interoperability [DEPRECATED]

  {"cudaGLMapBufferObject",         {"hipGLMapBufferObject__",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // Not equal to cuGLMapBufferObject due to different signatures
  {"cudaGLMapBufferObjectAsync",    {"hipGLMapBufferObjectAsync__",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // Not equal to cuGLMapBufferObjectAsync due to different signatures
  {"cudaGLRegisterBufferObject",    {"hipGLRegisterBufferObject",               CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLRegisterBufferObject)
  {"cudaGLSetBufferObjectMapFlags", {"hipGLSetBufferObjectMapFlags",            CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLSetBufferObjectMapFlags)
  {"cudaGLSetGLDevice",             {"hipGLSetGLDevice",                        CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
  {"cudaGLUnmapBufferObject",       {"hipGLUnmapBufferObject",                  CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnmapBufferObject)
  {"cudaGLUnmapBufferObjectAsync",  {"hipGLUnmapBufferObjectAsync",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnmapBufferObjectAsync)
  {"cudaGLUnregisterBufferObject",  {"hipGLUnregisterBufferObject",             CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnregisterBufferObject)

  // Direct3D 9 Interoperability

  {"cudaD3D9GetDevice",                {"hipD3D9GetDevice",                   CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDevice)
  {"cudaD3D9GetDevices",               {"hipD3D9GetDevices",                  CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDevices)
  {"cudaD3D9GetDirect3DDevice",        {"hipD3D9GetDirect3DDevice",           CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDirect3DDevice)
  {"cudaD3D9SetDirect3DDevice",        {"hipD3D9SetDirect3DDevice",           CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
  {"cudaGraphicsD3D9RegisterResource", {"hipGraphicsD3D9RegisterResource",    CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D9RegisterResource)

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

  {"cudaD3D10GetDevice",                {"hipD3D10GetDevice",                   CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10GetDevice)
  {"cudaD3D10GetDevices",               {"hipD3D10GetDevices",                  CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10GetDevices)
  {"cudaGraphicsD3D10RegisterResource", {"hipGraphicsD3D10RegisterResource",    CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D10RegisterResource)

  // Direct3D 10 Interoperability [DEPRECATED]

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
};
