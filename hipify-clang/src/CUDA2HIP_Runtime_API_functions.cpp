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

#include "CUDA2HIP.h"

// Map of all CUDA Runtime API functions
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_FUNCTION_MAP{
  // Error API
  {"cudaGetLastError",    {"hipGetLastError",    "", CONV_ERROR, API_RUNTIME}},
  {"cudaPeekAtLastError", {"hipPeekAtLastError", "", CONV_ERROR, API_RUNTIME}},
  {"cudaGetErrorName",    {"hipGetErrorName",    "", CONV_ERROR, API_RUNTIME}},
  {"cudaGetErrorString",  {"hipGetErrorString",  "", CONV_ERROR, API_RUNTIME}},

  // memcpy functions
  // no analogue
  // NOTE: Not equal to cuMemcpy due to different signatures
  {"cudaMemcpy",                 {"hipMemcpy",                 "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemcpyToArray",          {"hipMemcpyToArray",          "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemcpyToSymbol",         {"hipMemcpyToSymbol",         "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemcpyToSymbolAsync",    {"hipMemcpyToSymbolAsync",    "", CONV_MEMORY, API_RUNTIME}},

  {"cudaMemcpyAsync",            {"hipMemcpyAsync",            "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMemcpy2D due to different signatures
  {"cudaMemcpy2D",               {"hipMemcpy2D",               "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMemcpy2DAsync due to different signatures
  {"cudaMemcpy2DAsync",          {"hipMemcpy2DAsync",          "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemcpy2DToArray",        {"hipMemcpy2DToArray",        "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemcpy2DArrayToArray",   {"hipMemcpy2DArrayToArray",   "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaMemcpy2DFromArray",      {"hipMemcpy2DFromArray",      "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaMemcpy2DFromArrayAsync", {"hipMemcpy2DFromArrayAsync", "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaMemcpy2DToArrayAsync",   {"hipMemcpy2DToArrayAsync",   "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3D due to different signatures
  {"cudaMemcpy3D",               {"hipMemcpy3D",               "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DAsync due to different signatures
  {"cudaMemcpy3DAsync",          {"hipMemcpy3DAsync",          "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeer due to different signatures
  {"cudaMemcpy3DPeer",           {"hipMemcpy3DPeer",           "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeerAsync due to different signatures
  {"cudaMemcpy3DPeerAsync",      {"hipMemcpy3DPeerAsync",      "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpyAtoA due to different signatures
  {"cudaMemcpyArrayToArray",     {"hipMemcpyArrayToArray",     "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaMemcpyFromArrayAsync",   {"hipMemcpyFromArrayAsync",   "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaMemcpyFromSymbol",       {"hipMemcpyFromSymbol",       "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemcpyFromSymbolAsync",  {"hipMemcpyFromSymbolAsync",  "", CONV_MEMORY, API_RUNTIME}},
  // cuMemAdvise
  {"cudaMemAdvise",              {"hipMemAdvise",              "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuMemRangeGetAttribute
  {"cudaMemRangeGetAttribute",   {"hipMemRangeGetAttribute",   "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuMemRangeGetAttributes
  {"cudaMemRangeGetAttributes",  {"hipMemRangeGetAttributes",  "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},

  // memset
  {"cudaMemset",        {"hipMemset",        "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemsetAsync",   {"hipMemsetAsync",   "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemset2D",      {"hipMemset2D",      "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemset2DAsync", {"hipMemset2DAsync", "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMemset3D",      {"hipMemset3D",      "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaMemset3DAsync", {"hipMemset3DAsync", "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},

  // Memory management
  // cuMemGetInfo
  {"cudaMemGetInfo",             {"hipMemGetInfo",             "", CONV_MEMORY, API_RUNTIME}},
  {"cudaArrayGetInfo",           {"hipArrayGetInfo",           "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayDestroy due to different signatures
  {"cudaFreeMipmappedArray",     {"hipFreeMipmappedArray",     "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGetMipmappedArrayLevel", {"hipGetMipmappedArrayLevel", "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGetSymbolAddress",       {"hipGetSymbolAddress",       "", CONV_MEMORY, API_RUNTIME}},
  {"cudaGetSymbolSize",          {"hipGetSymbolSize",          "", CONV_MEMORY, API_RUNTIME}},
  // TODO: double check cuMemPrefetchAsync
  {"cudaMemPrefetchAsync",       {"hipMemPrefetchAsync",       "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},

  // malloc
  {"cudaMalloc",               {"hipMalloc",               "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMallocHost",           {"hipHostMalloc",           "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMallocArray",          {"hipMallocArray",          "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMalloc3D",             {"hipMalloc3D",             "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMalloc3DArray",        {"hipMalloc3DArray",        "", CONV_MEMORY, API_RUNTIME}},
  {"cudaMallocManaged",        {"hipMallocManaged",        "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayCreate due to different signatures
  {"cudaMallocMipmappedArray", {"hipMallocMipmappedArray", "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaMallocPitch",          {"hipMallocPitch",          "", CONV_MEMORY, API_RUNTIME}},

  // cuMemFree
  {"cudaFree",           {"hipFree",           "", CONV_MEMORY, API_RUNTIME}},
  // cuMemFreeHost
  {"cudaFreeHost",       {"hipHostFree",       "", CONV_MEMORY, API_RUNTIME}},
  {"cudaFreeArray",      {"hipFreeArray",      "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostRegister
  {"cudaHostRegister",   {"hipHostRegister",   "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostUnregister
  {"cudaHostUnregister", {"hipHostUnregister", "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostAlloc
  // NOTE: hipHostAlloc deprecated - use hipHostMalloc instead
  {"cudaHostAlloc",      {"hipHostMalloc",     "", CONV_MEMORY, API_RUNTIME}},

  // make memory functions
  {"make_cudaExtent",     {"make_hipExtent",     "", CONV_MEMORY, API_RUNTIME}},
  {"make_cudaPitchedPtr", {"make_hipPitchedPtr", "", CONV_MEMORY, API_RUNTIME}},
  {"make_cudaPos",        {"make_hipPos",        "", CONV_MEMORY, API_RUNTIME}},

  // Host Register Flags
  // cuMemHostGetFlags
  {"cudaHostGetFlags",         {"hipHostGetFlags",         "", CONV_MEMORY, API_RUNTIME}},

  // Events
  // no analogue
  // NOTE: Not equal to cuEventCreate due to different signatures
  {"cudaEventCreate",              {"hipEventCreate",              "", CONV_EVENT, API_RUNTIME}},
  // cuEventCreate
  {"cudaEventCreateWithFlags",     {"hipEventCreateWithFlags",     "", CONV_EVENT, API_RUNTIME}},
  // cuEventDestroy
  {"cudaEventDestroy",             {"hipEventDestroy",             "", CONV_EVENT, API_RUNTIME}},
  // cuEventRecord
  {"cudaEventRecord",              {"hipEventRecord",              "", CONV_EVENT, API_RUNTIME}},
  // cuEventElapsedTime
  {"cudaEventElapsedTime",         {"hipEventElapsedTime",         "", CONV_EVENT, API_RUNTIME}},
  // cuEventSynchronize
  {"cudaEventSynchronize",         {"hipEventSynchronize",         "", CONV_EVENT, API_RUNTIME}},
  // cuEventQuery
  {"cudaEventQuery",               {"hipEventQuery",               "", CONV_EVENT, API_RUNTIME}},

  // 5.6. External Resource Interoperability
  // cuDestroyExternalMemory
  {"cudaDestroyExternalMemory",                            {"hipDestroyExternalMemory",                                "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuDestroyExternalSemaphore
  {"cudaDestroyExternalSemaphore",                         {"hipDestroyExternalSemaphore",                             "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuExternalMemoryGetMappedBuffer
  {"cudaExternalMemoryGetMappedBuffer",                    {"hipExternalMemoryGetMappedBuffer",                        "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuExternalMemoryGetMappedMipmappedArray
  {"cudaExternalMemoryGetMappedMipmappedArray",            {"hipExternalMemoryGetMappedMipmappedArray",                "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuImportExternalMemory
  {"cudaImportExternalMemory",                             {"hipImportExternalMemory",                                 "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuImportExternalSemaphore
  {"cudaImportExternalSemaphore",                          {"hipImportExternalSemaphore",                              "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuSignalExternalSemaphoresAsync
  {"cudaSignalExternalSemaphoresAsync",                    {"hipSignalExternalSemaphoresAsync",                        "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuWaitExternalSemaphoresAsync
  {"cudaWaitExternalSemaphoresAsync",                      {"hipWaitExternalSemaphoresAsync",                          "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},

  // Streams
  // no analogue
  // NOTE: Not equal to cuStreamCreate due to different signatures
  {"cudaStreamCreate",             {"hipStreamCreate",             "", CONV_STREAM, API_RUNTIME}},
  // cuStreamCreate
  {"cudaStreamCreateWithFlags",    {"hipStreamCreateWithFlags",    "", CONV_STREAM, API_RUNTIME}},
  // cuStreamCreateWithPriority
  {"cudaStreamCreateWithPriority", {"hipStreamCreateWithPriority", "", CONV_STREAM, API_RUNTIME}},
  // cuStreamDestroy
  {"cudaStreamDestroy",            {"hipStreamDestroy",            "", CONV_STREAM, API_RUNTIME}},
  // cuStreamWaitEvent
  {"cudaStreamWaitEvent",          {"hipStreamWaitEvent",          "", CONV_STREAM, API_RUNTIME}},
  // cuStreamSynchronize
  {"cudaStreamSynchronize",        {"hipStreamSynchronize",        "", CONV_STREAM, API_RUNTIME}},
  // cuStreamGetFlags
  {"cudaStreamGetFlags",           {"hipStreamGetFlags",           "", CONV_STREAM, API_RUNTIME}},
  // cuStreamQuery
  {"cudaStreamQuery",              {"hipStreamQuery",              "", CONV_STREAM, API_RUNTIME}},
  // cuStreamAddCallback
  {"cudaStreamAddCallback",        {"hipStreamAddCallback",        "", CONV_STREAM, API_RUNTIME}},
  // cuStreamAttachMemAsync
  {"cudaStreamAttachMemAsync",     {"hipStreamAttachMemAsync",     "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamBeginCapture
  {"cudaStreamBeginCapture",       {"hipStreamBeginCapture",       "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamEndCapture
  {"cudaStreamEndCapture",         {"hipStreamEndCapture",         "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamIsCapturing
  {"cudaStreamIsCapturing",        {"hipStreamIsCapturing",        "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamGetPriority
  {"cudaStreamGetPriority",        {"hipStreamGetPriority",        "", CONV_STREAM, API_RUNTIME}},

  // Other synchronization
  {"cudaDeviceSynchronize", {"hipDeviceSynchronize", "", CONV_DEVICE, API_RUNTIME}},
  {"cudaDeviceReset",       {"hipDeviceReset",       "", CONV_DEVICE, API_RUNTIME}},
  {"cudaSetDevice",         {"hipSetDevice",         "", CONV_DEVICE, API_RUNTIME}},
  {"cudaGetDevice",         {"hipGetDevice",         "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGetCount
  {"cudaGetDeviceCount",    {"hipGetDeviceCount",    "", CONV_DEVICE, API_RUNTIME}},
  {"cudaChooseDevice",      {"hipChooseDevice",      "", CONV_DEVICE, API_RUNTIME}},

  // Thread Management
  {"cudaThreadExit",           {"hipDeviceReset",          "", CONV_THREAD, API_RUNTIME}},
  {"cudaThreadGetCacheConfig", {"hipDeviceGetCacheConfig", "", CONV_THREAD, API_RUNTIME}},
  {"cudaThreadGetLimit",       {"hipThreadGetLimit",       "", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaThreadSetCacheConfig", {"hipDeviceSetCacheConfig", "", CONV_THREAD, API_RUNTIME}},
  {"cudaThreadSetLimit",       {"hipThreadSetLimit",       "", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaThreadSynchronize",    {"hipDeviceSynchronize",    "", CONV_THREAD, API_RUNTIME}},

  // Attributes
  {"cudaDeviceGetAttribute",                      {"hipDeviceGetAttribute",                              "", CONV_DEVICE, API_RUNTIME}},

  // Pointer Attributes
  // no analogue
  // NOTE: Not equal to cuPointerGetAttributes due to different signatures
  {"cudaPointerGetAttributes", {"hipPointerGetAttributes", "", CONV_ADDRESSING, API_RUNTIME}},
  // cuMemHostGetDevicePointer
  {"cudaHostGetDevicePointer", {"hipHostGetDevicePointer", "", CONV_MEMORY,  API_RUNTIME}},

  // Device
  {"cudaGetDeviceProperties",          {"hipGetDeviceProperties",          "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGetPCIBusId
  {"cudaDeviceGetPCIBusId",            {"hipDeviceGetPCIBusId",            "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGetByPCIBusId
  {"cudaDeviceGetByPCIBusId",          {"hipDeviceGetByPCIBusId",          "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxGetStreamPriorityRange
  {"cudaDeviceGetStreamPriorityRange", {"hipDeviceGetStreamPriorityRange", "", CONV_DEVICE, API_RUNTIME}},
  {"cudaSetValidDevices",              {"hipSetValidDevices",              "", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},

  // Device Flags
  // cuCtxGetFlags
  {"cudaGetDeviceFlags", {"hipCtxGetFlags", "", CONV_DEVICE, API_RUNTIME}},
  {"cudaSetDeviceFlags", {"hipSetDeviceFlags", "", CONV_DEVICE, API_RUNTIME}},

  // Cache config
  {"cudaDeviceSetCacheConfig", {"hipDeviceSetCacheConfig", "", CONV_DEVICE, API_RUNTIME}},
  {"cudaDeviceGetCacheConfig", {"hipDeviceGetCacheConfig", "", CONV_DEVICE, API_RUNTIME}},
  {"cudaFuncSetCacheConfig",   {"hipFuncSetCacheConfig",   "", CONV_DEVICE, API_RUNTIME}},


  // Execution control functions
  {"cudaFuncGetAttributes",      {"hipFuncGetAttributes",      "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaFuncSetSharedMemConfig", {"hipFuncSetSharedMemConfig", "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGetParameterBuffer",     {"hipGetParameterBuffer",     "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaSetDoubleForDevice",     {"hipSetDoubleForDevice",     "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaSetDoubleForHost",       {"hipSetDoubleForHost",       "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},

  // Execution Control [deprecated since 7.0]
  {"cudaConfigureCall", {"hipConfigureCall", "", CONV_EXECUTION, API_RUNTIME}},
  {"cudaLaunch",        {"hipLaunchByPtr",   "", CONV_EXECUTION, API_RUNTIME}},
  {"cudaSetupArgument", {"hipSetupArgument", "", CONV_EXECUTION, API_RUNTIME}},

  // Version Management
  {"cudaDriverGetVersion",  {"hipDriverGetVersion",  "", CONV_VERSION, API_RUNTIME}},
  {"cudaRuntimeGetVersion", {"hipRuntimeGetVersion", "", CONV_VERSION, API_RUNTIME, HIP_UNSUPPORTED}},

  // Occupancy
  {"cudaOccupancyMaxPotentialBlockSize",                      {"hipOccupancyMaxPotentialBlockSize",                      "", CONV_OCCUPANCY, API_RUNTIME}},
  {"cudaOccupancyMaxPotentialBlockSizeWithFlags",             {"hipOccupancyMaxPotentialBlockSizeWithFlags",             "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaOccupancyMaxActiveBlocksPerMultiprocessor",           {"hipOccupancyMaxActiveBlocksPerMultiprocessor",           "", CONV_OCCUPANCY, API_RUNTIME}},
  {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaOccupancyMaxPotentialBlockSizeVariableSMem",          {"hipOccupancyMaxPotentialBlockSizeVariableSMem",          "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", {"hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},

  // Peer2Peer
  {"cudaDeviceCanAccessPeer",     {"hipDeviceCanAccessPeer",     "", CONV_PEER, API_RUNTIME}},
  {"cudaDeviceDisablePeerAccess", {"hipDeviceDisablePeerAccess", "", CONV_PEER, API_RUNTIME}},
  {"cudaDeviceEnablePeerAccess",  {"hipDeviceEnablePeerAccess",  "", CONV_PEER, API_RUNTIME}},

  {"cudaMemcpyPeerAsync",         {"hipMemcpyPeerAsync",         "", CONV_MEMORY,  API_RUNTIME}},
  {"cudaMemcpyPeer",              {"hipMemcpyPeer",              "", CONV_MEMORY,  API_RUNTIME}},

  // Shared memory
  {"cudaDeviceSetSharedMemConfig",   {"hipDeviceSetSharedMemConfig",   "", CONV_DEVICE, API_RUNTIME}},
  {"cudaDeviceGetSharedMemConfig",   {"hipDeviceGetSharedMemConfig",   "", CONV_DEVICE, API_RUNTIME}},
  // translate deprecated
  //     {"cudaThreadGetSharedMemConfig", {"hipDeviceGetSharedMemConfig", "", CONV_DEVICE, API_RUNTIME}},
  //     {"cudaThreadSetSharedMemConfig", {"hipDeviceSetSharedMemConfig", "", CONV_DEVICE, API_RUNTIME}},

  // cuCtxGetLimit
  {"cudaDeviceGetLimit",                    {"hipDeviceGetLimit",                    "", CONV_DEVICE, API_RUNTIME}},

  // Profiler
  {"cudaProfilerInitialize", {"hipProfilerInitialize", "", CONV_PROFILER, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuProfilerInitialize)
  {"cudaProfilerStart",      {"hipProfilerStart",      "", CONV_PROFILER, API_RUNTIME}},                     // API_Driver ANALOGUE (cuProfilerStart)
  {"cudaProfilerStop",       {"hipProfilerStop",       "", CONV_PROFILER, API_RUNTIME}},                     // API_Driver ANALOGUE (cuProfilerStop)


  {"cudaBindTexture",                 {"hipBindTexture",                 "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaUnbindTexture",               {"hipUnbindTexture",               "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaBindTexture2D",               {"hipBindTexture2D",               "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaBindTextureToArray",          {"hipBindTextureToArray",          "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaBindTextureToMipmappedArray", {"hipBindTextureToMipmappedArray", "", CONV_TEXTURE, API_RUNTIME}},    // Unsupported yet on NVCC path
  {"cudaGetTextureAlignmentOffset",   {"hipGetTextureAlignmentOffset",   "", CONV_TEXTURE, API_RUNTIME}},    // Unsupported yet on NVCC path
  {"cudaGetTextureReference",         {"hipGetTextureReference",         "", CONV_TEXTURE, API_RUNTIME}},    // Unsupported yet on NVCC path

  {"cudaCreateChannelDesc",         {"hipCreateChannelDesc",         "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaGetChannelDesc",            {"hipGetChannelDesc",            "", CONV_TEXTURE, API_RUNTIME}},

  // Texture Object Management

  {"cudaAddressModeWrap",    {"hipAddressModeWrap",    "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaAddressModeClamp",   {"hipAddressModeClamp",   "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaAddressModeMirror",  {"hipAddressModeMirror",  "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaAddressModeBorder",  {"hipAddressModeBorder",  "", CONV_TEXTURE, API_RUNTIME}},

  // functions
  {"cudaCreateTextureObject",              {"hipCreateTextureObject",              "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaDestroyTextureObject",             {"hipDestroyTextureObject",             "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaGetTextureObjectResourceDesc",     {"hipGetTextureObjectResourceDesc",     "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaGetTextureObjectResourceViewDesc", {"hipGetTextureObjectResourceViewDesc", "", CONV_TEXTURE, API_RUNTIME}},
  {"cudaGetTextureObjectTextureDesc",      {"hipGetTextureObjectTextureDesc",      "", CONV_TEXTURE, API_RUNTIME}},

  // Surface Reference Management
  {"cudaBindSurfaceToArray",  {"hipBindSurfaceToArray",  "", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGetSurfaceReference", {"hipGetSurfaceReference", "", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},


  // Surface Object Management
  {"cudaCreateSurfaceObject",          {"hipCreateSurfaceObject",          "", CONV_SURFACE, API_RUNTIME}},
  {"cudaDestroySurfaceObject",         {"hipDestroySurfaceObject",         "", CONV_SURFACE, API_RUNTIME}},
  {"cudaGetSurfaceObjectResourceDesc", {"hipGetSurfaceObjectResourceDesc", "", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

  // Inter-Process Communications (IPC)
  {"cudaIpcCloseMemHandle",  {"hipIpcCloseMemHandle",  "", CONV_DEVICE, API_RUNTIME}},
  {"cudaIpcGetEventHandle",  {"hipIpcGetEventHandle",  "", CONV_DEVICE, API_RUNTIME}},
  {"cudaIpcGetMemHandle",    {"hipIpcGetMemHandle",    "", CONV_DEVICE, API_RUNTIME}},
  {"cudaIpcOpenEventHandle", {"hipIpcOpenEventHandle", "", CONV_DEVICE, API_RUNTIME}},
  {"cudaIpcOpenMemHandle",   {"hipIpcOpenMemHandle",   "", CONV_DEVICE, API_RUNTIME}},

  // OpenGL Interoperability
  {"cudaGLGetDevices",             {"hipGLGetDevices",             "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsGLRegisterBuffer", {"hipGraphicsGLRegisterBuffer", "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsGLRegisterImage",  {"hipGraphicsGLRegisterImage",  "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaWGLGetDevice",             {"hipWGLGetDevice",             "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},

  // Graphics Interoperability
  {"cudaGraphicsMapResources",                    {"hipGraphicsMapResources",                    "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsMapResources)
  {"cudaGraphicsResourceGetMappedMipmappedArray", {"hipGraphicsResourceGetMappedMipmappedArray", "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceGetMappedMipmappedArray)
  {"cudaGraphicsResourceGetMappedPointer",        {"hipGraphicsResourceGetMappedPointer",        "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceGetMappedPointer)
  {"cudaGraphicsResourceSetMapFlags",             {"hipGraphicsResourceSetMapFlags",             "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceSetMapFlags)
  {"cudaGraphicsSubResourceGetMappedArray",       {"hipGraphicsSubResourceGetMappedArray",       "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsSubResourceGetMappedArray)
  {"cudaGraphicsUnmapResources",                  {"hipGraphicsUnmapResources",                  "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsUnmapResources)
  {"cudaGraphicsUnregisterResource",              {"hipGraphicsUnregisterResource",              "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsUnregisterResource)

  {"cudaGLGetDevices",             {"hipGLGetDevices",                  "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLGetDevices)
  {"cudaGraphicsGLRegisterBuffer", {"hipGraphicsGLRegisterBuffer",      "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsGLRegisterBuffer)
  {"cudaGraphicsGLRegisterImage",  {"hipGraphicsGLRegisterImage",       "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsGLRegisterImage)
  {"cudaWGLGetDevice",             {"hipWGLGetDevice",                  "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuWGLGetDevice)

  // OpenGL Interoperability [DEPRECATED]

  {"cudaGLMapBufferObject",         {"hipGLMapBufferObject__",                  "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // Not equal to cuGLMapBufferObject due to different signatures
  {"cudaGLMapBufferObjectAsync",    {"hipGLMapBufferObjectAsync__",             "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // Not equal to cuGLMapBufferObjectAsync due to different signatures
  {"cudaGLRegisterBufferObject",    {"hipGLRegisterBufferObject",               "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLRegisterBufferObject)
  {"cudaGLSetBufferObjectMapFlags", {"hipGLSetBufferObjectMapFlags",            "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLSetBufferObjectMapFlags)
  {"cudaGLSetGLDevice",             {"hipGLSetGLDevice",                        "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
  {"cudaGLUnmapBufferObject",       {"hipGLUnmapBufferObject",                  "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnmapBufferObject)
  {"cudaGLUnmapBufferObjectAsync",  {"hipGLUnmapBufferObjectAsync",             "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnmapBufferObjectAsync)
  {"cudaGLUnregisterBufferObject",  {"hipGLUnregisterBufferObject",             "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGLUnregisterBufferObject)

  // Direct3D 9 Interoperability

  {"cudaD3D9GetDevice",                {"hipD3D9GetDevice",                   "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDevice)
  {"cudaD3D9GetDevices",               {"hipD3D9GetDevices",                  "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDevices)
  {"cudaD3D9GetDirect3DDevice",        {"hipD3D9GetDirect3DDevice",           "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9GetDirect3DDevice)
  {"cudaD3D9SetDirect3DDevice",        {"hipD3D9SetDirect3DDevice",           "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
  {"cudaGraphicsD3D9RegisterResource", {"hipGraphicsD3D9RegisterResource",    "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D9RegisterResource)

  {"cudaD3D9MapResources",                 {"hipD3D9MapResources",                 "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9MapResources)
  {"cudaD3D9RegisterResource",             {"hipD3D9RegisterResource",             "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9RegisterResource)
  {"cudaD3D9ResourceGetMappedArray",       {"hipD3D9ResourceGetMappedArray",       "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetMappedArray)
  {"cudaD3D9ResourceGetMappedPitch",       {"hipD3D9ResourceGetMappedPitch",       "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cudaD3D9ResourceGetMappedPitch)
  {"cudaD3D9ResourceGetMappedPointer",     {"hipD3D9ResourceGetMappedPointer",     "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetMappedPointer)
  {"cudaD3D9ResourceGetMappedSize",        {"hipD3D9ResourceGetMappedSize",        "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetMappedSize)
  {"cudaD3D9ResourceGetSurfaceDimensions", {"hipD3D9ResourceGetSurfaceDimensions", "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceGetSurfaceDimensions)
  {"cudaD3D9ResourceSetMapFlags",          {"hipD3D9ResourceSetMapFlags",          "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9ResourceSetMapFlags)
  {"cudaD3D9UnmapResources",               {"hipD3D9UnmapResources",               "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9UnmapResources)
  {"cudaD3D9UnregisterResource",           {"hipD3D9UnregisterResource",           "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D9UnregisterResource)

  // Direct3D 10 Interoperability

  {"cudaD3D10GetDevice",                {"hipD3D10GetDevice",                   "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10GetDevice)
  {"cudaD3D10GetDevices",               {"hipD3D10GetDevices",                  "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10GetDevices)
  {"cudaGraphicsD3D10RegisterResource", {"hipGraphicsD3D10RegisterResource",    "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D10RegisterResource)

  // Direct3D 10 Interoperability [DEPRECATED]

  {"cudaD3D10GetDirect3DDevice",            {"hipD3D10GetDirect3DDevice",                "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cudaD3D10GetDirect3DDevice)
  {"cudaD3D10MapResources",                 {"hipD3D10MapResources",                     "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10MapResources)
  {"cudaD3D10RegisterResource",             {"hipD3D10RegisterResource",                 "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10RegisterResource)
  {"cudaD3D10ResourceGetMappedArray",       {"hipD3D10ResourceGetMappedArray",           "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetMappedArray)
  {"cudaD3D10ResourceGetMappedPitch",       {"hipD3D10ResourceGetMappedPitch",           "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cudaD3D10ResourceGetMappedPitch)
  {"cudaD3D10ResourceGetMappedPointer",     {"hipD3D10ResourceGetMappedPointer",         "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetMappedPointer)
  {"cudaD3D10ResourceGetMappedSize",        {"hipD3D10ResourceGetMappedSize",            "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetMappedSize)
  {"cudaD3D10ResourceGetSurfaceDimensions", {"hipD3D10ResourceGetSurfaceDimensions",     "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceGetSurfaceDimensions)
  {"cudaD3D10ResourceSetMapFlags",          {"hipD3D10ResourceSetMapFlags",              "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10ResourceSetMapFlags)
  {"cudaD3D10SetDirect3DDevice",            {"hipD3D10SetDirect3DDevice",                "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE
  {"cudaD3D10UnmapResources",               {"hipD3D10UnmapResources",                   "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10UnmapResources)
  {"cudaD3D10UnregisterResource",           {"hipD3D10UnregisterResource",               "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D10UnregisterResource)

  // Direct3D 11 Interoperability

  {"cudaD3D11GetDevice",                {"hipD3D11GetDevice",                   "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevice)
  {"cudaD3D11GetDevices",               {"hipD3D11GetDevices",                  "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevices)
  {"cudaGraphicsD3D11RegisterResource", {"hipGraphicsD3D11RegisterResource",    "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D11RegisterResource)

  // Direct3D 11 Interoperability [DEPRECATED]
  {"cudaD3D11GetDevice",                     {"hipD3D11GetDevice",                     "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevice)
  {"cudaD3D11GetDevices",                    {"hipD3D11GetDevices",                    "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuD3D11GetDevices)
  {"cudaGraphicsD3D11RegisterResource",      {"hipGraphicsD3D11RegisterResource",      "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsD3D11RegisterResource)

  // VDPAU Interoperability
  {"cudaGraphicsVDPAURegisterOutputSurface", {"hipGraphicsVDPAURegisterOutputSurface", "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsVDPAURegisterOutputSurface)
  {"cudaGraphicsVDPAURegisterVideoSurface",  {"hipGraphicsVDPAURegisterVideoSurface",  "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsVDPAURegisterVideoSurface)
  {"cudaVDPAUGetDevice",                     {"hipVDPAUGetDevice",                     "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuVDPAUGetDevice)
  {"cudaVDPAUSetVDPAUDevice",                {"hipVDPAUSetDevice",                     "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},    // no API_Driver ANALOGUE

  // EGL Interoperability
  {"cudaEGLStreamConsumerAcquireFrame",     {"hipEGLStreamConsumerAcquireFrame",     "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerAcquireFrame)
  {"cudaEGLStreamConsumerConnect",          {"hipEGLStreamConsumerConnect",          "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerConnect)
  {"cudaEGLStreamConsumerConnectWithFlags", {"hipEGLStreamConsumerConnectWithFlags", "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerConnectWithFlags)
  {"cudaEGLStreamConsumerReleaseFrame",     {"hipEGLStreamConsumerReleaseFrame",     "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamConsumerReleaseFrame)
  {"cudaEGLStreamProducerConnect",          {"hipEGLStreamProducerConnect",          "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerConnect)
  {"cudaEGLStreamProducerDisconnect",       {"hipEGLStreamProducerDisconnect",       "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerDisconnect)
  {"cudaEGLStreamProducerPresentFrame",     {"hipEGLStreamProducerPresentFrame",     "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerPresentFrame)
  {"cudaEGLStreamProducerReturnFrame",      {"hipEGLStreamProducerReturnFrame",      "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuEGLStreamProducerReturnFrame)
  {"cudaGraphicsEGLRegisterImage",          {"hipGraphicsEGLRegisterImage",          "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsEGLRegisterImage)
  {"cudaGraphicsResourceGetMappedEglFrame", {"hipGraphicsResourceGetMappedEglFrame", "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (cuGraphicsResourceGetMappedEglFrame)
};
